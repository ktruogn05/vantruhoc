# Đặc tả Môi trường Giả lập: LLM Serving Simulator Environment (`LLMEnv`)

Tài liệu này định nghĩa chi tiết cơ chế vận hành vật lý, các ràng buộc hệ thống, không gian trạng thái, không gian hành động và cách thức chuyển đổi trạng thái của Môi trường Giả lập phục vụ LLM (`LLMEnv`). Môi trường này mô phỏng các hoạt động thực tế của một thực thể phục vụ (**GPU Worker Instance**) sử dụng các kỹ thuật **Continuous Batching** và **PagedAttention**.

---

## 1. Trạng thái của Môi trường (State Representation)

Tại mỗi bước thời gian mô phỏng $t$ (tương ứng với 1 chu kỳ sinh token/iteration), trạng thái bên trong của môi trường được định nghĩa bởi bộ dữ liệu sau:

### A. Quản lý Bộ nhớ (KV Cache Memory Pool)
* $N_{blocks} = 1000$: Tổng số block KV Cache vật lý tối đa trên VRAM GPU (Hằng số hệ thống).
* $B_{size} = 16$: Số lượng token tối đa mà một block có thể chứa (Hằng số hệ thống).
* `free_blocks`: Số lượng block vật lý còn trống trong VRAM ($0 \le \text{free\_blocks} \le N_{blocks}$).
* `block_table`: Bảng băm ánh xạ từ `request_id` sang một danh sách các ID block vật lý đang cấp phát trên GPU:
  $$\text{block\_table} = \{ R_i \to [b_1, b_2, \dots, b_k] \}$$
* $N_{cpu\_blocks} = 2000$: Tổng số block KV Cache tối đa có thể swap trên bộ nhớ RAM của máy chủ (CPU host RAM).
* `swapped_blocks`: Số lượng block KV Cache ảo đang lưu trên RAM CPU của các request bị tạm dừng.

### B. Danh sách Batch đang chạy (`active_batch`)
Tập hợp các request đang được xử lý đồng thời trên GPU tại step $t$ ($\text{len}(active\_batch) \le Batch_{max} = 16$). Với mỗi request $i \in \text{active\_batch}$, môi trường lưu giữ cấu trúc dữ liệu sau:
* `stage`: Trạng thái xử lý của request:
  * `PREFILL`: Đang trong pha prefill xử lý prompt đầu vào.
  * `SWAP_IN_DEGRADED`: Đang trong pha chuyển cache từ CPU về GPU, tốn $K_{swap} = 3$ steps.
  * `DECODE`: Đang trong pha sinh token từng bước.
* `swap_in_remaining`: Số bước thời gian còn lại cần để hoàn thành quá trình swap-in (chỉ áp dụng khi `stage == SWAP_IN_DEGRADED`).
* `prompt_tokens`: Độ dài prompt đầu vào ($p_i$).
* `tokens_generated`: Số lượng token response đã sinh ra đến thời điểm hiện tại.
* `allocated_blocks`: Danh sách các ID block vật lý được cấp phát trên GPU.
* `allocated_blocks_cpu`: Số lượng block KV Cache đang chiếm dụng trên CPU RAM (chỉ lưu giá trị khi bị swap, mặc định = 0).
* `priority`: Mức độ ưu tiên của request ($pr_i \in \{1, 2, 3\}$).
* `arrival_time`: Thời điểm request đến hệ thống ($a_i$).
* `deadline`: Thời điểm tối đa phải hoàn thành request ($d_i$).

### C. Các Hàng đợi Trạng thái (Queues)
* `queue` (Hàng đợi chờ): Chứa các request mới đến hệ thống trực tuyến nhưng chưa được nạp vào batch lần nào. Các request được sắp xếp theo thời điểm đến $a_i$.
* `preempted_queue` (Hàng đợi tạm dừng): Chứa các request đang chạy dở nhưng bị tạm dừng để giải phóng VRAM. Mỗi request trong hàng đợi này mang một trong hai trạng thái:
  * `RECOMPUTE_WAITING`: Request bị giải phóng hoàn toàn bộ nhớ. Khi khôi phục sẽ phải chạy lại pha Prefill từ đầu.
  * `SWAPPED_TO_CPU`: Request được lưu trữ KV Cache trên RAM CPU. Khi khôi phục chỉ cần nạp ngược lại VRAM GPU nhưng chịu độ trễ swap-in.

### D. Trục thời gian hệ thống
* `current_time` ($t$): Thời gian mô phỏng hiện tại, bắt đầu từ $t = 0$. Tăng tuyến tính từng bước $+1$ sau mỗi chu kỳ.

---

## 2. Quy trình chuyển trạng thái nghiêm ngặt (`env.step(action)`)

Mỗi khi nhận được một hành động (`action`) gửi đến từ Bộ lập lịch tại đầu step $t$, môi trường `LLMEnv` sẽ thực thi quá trình chuyển trạng thái qua **4 pha tuần tự** sau:

### Pha 1: Thực thi Hành động Lập lịch (Process Scheduler Actions)
Hệ thống xử lý lệnh cấu trúc lại batch từ Bộ lập lịch (chỉ cho phép một hành động duy nhất tại mỗi step):
1. **Lệnh `Promote(request_ids: List[str])`**:
   * Hệ thống duyệt qua danh sách `request_ids` được chọn (tối đa 2 requests để tránh quá tải Prefill GPU):
     * *Tính toán số block prefill cơ sở*: $B_{req} = \lceil p_i / B_{size} \rceil$.
     * *Kiểm tra điều kiện nạp*: Nếu batch chưa đầy ($\text{len}(active\_batch) < Batch_{max}$), số lượng prefill được thêm vào không vượt quá giới hạn prefill song song ở step này (tổng prefill $\le 2$), và GPU có đủ bộ nhớ: $free\_blocks \ge B_{req}$:
       * Thực hiện trừ bộ nhớ trống: $free\_blocks \leftarrow free\_blocks - B_{req}$.
       * Cấp phát $B_{req}$ blocks vật lý trống từ GPU cho request $i$, lưu danh sách ID này vào `allocated_blocks` của request và map vào `block_table[request_id]`.
       * Đưa request $i$ vào `active_batch` với `stage = PREFILL` và `tokens_generated = 0`.
       * Xóa request $i$ khỏi `queue`.
     * Nếu không thỏa mãn điều kiện: Từ chối nạp request đó và giữ nguyên ở `queue`.
2. **Lệnh `Preempt(request_ids: List[str], strategy: str)`**:
   * Với từng `request_id` ($i$) trong danh sách `request_ids` được chỉ định:
     * *Kiểm tra tính nguyên tử (Atomicity)*: Nếu request $i$ đang ở `stage == PREFILL`, hệ thống **từ chối preempt** đối với request này để bảo vệ tính nguyên tử tuyệt đối của pha prefill.
     * Xóa request $i$ khỏi `active_batch` và `block_table`.
     * **Nếu request đang ở trạng thái `stage == SWAP_IN_DEGRADED`**:
       * *Đặc biệt*: Vì KV Cache cũ của request này vẫn đang được lưu trữ nguyên vẹn trên CPU RAM (chưa được giải phóng thực tế vì chưa hoàn tất swap-in), việc preempt chỉ cần:
         * Thu hồi VRAM GPU đã cấp phát: $free\_blocks \leftarrow free\_blocks + \text{len}(allocated\_blocks)$.
         * Nếu `strategy == Recompute`: Giải phóng cả CPU RAM: $swapped\_blocks \leftarrow swapped\_blocks - request.allocated\_blocks\_cpu$. Đưa request vào `preempted_queue` với nhãn `RECOMPUTE_WAITING`.
         * Nếu `strategy == Swap`: Giữ nguyên CPU RAM. Đưa request ngược lại `preempted_queue` mang nhãn `SWAPPED_TO_CPU` (không cấp phát thêm CPU RAM mới vì cache cũ của nó trên CPU vẫn đang được giữ an toàn).
     * **Nếu request ở trạng thái bình thường (`stage == DECODE`)**:
       * **Nếu `strategy == Recompute`**:
         * Thu hồi toàn bộ block vật lý về bể chứa: $free\_blocks \leftarrow free\_blocks + \text{len}(allocated\_blocks)$.
         * Reset số token đã sinh: `tokens_generated = 0`.
         * Đưa request $i$ vào `preempted_queue` với trạng thái `RECOMPUTE_WAITING`.
       * **Nếu `strategy == Swap`**:
         * Kiểm tra dung lượng CPU RAM: Nếu $swapped\_blocks + \text{len}(allocated\_blocks) > N_{cpu\_blocks}$, hệ thống lập tức báo lỗi **Host OOM Crash** và dừng mô phỏng.
         * Thu hồi block vật lý trên GPU VRAM: $free\_blocks \leftarrow free\_blocks + \text{len}(allocated\_blocks)$.
         * Ghi nhận dung lượng chiếm dụng CPU RAM: $request.allocated\_blocks\_cpu = \text{len}(allocated\_blocks)$.
         * Lưu trữ trên bộ nhớ CPU Host: $swapped\_blocks \leftarrow swapped\_blocks + \text{len}(allocated\_blocks)$.
         * Giữ nguyên số token đã sinh: `tokens_generated` không đổi.
         * Đưa request $i$ vào `preempted_queue` với trạng thái `SWAPPED_TO_CPU`.
3. **Lệnh `Resume(request_ids: List[str])`**:
   * Hệ thống duyệt qua từng `request_id` ($i$) trong danh sách `request_ids` (tối đa 2 requests để tránh quá tải GPU):
     * Kiểm tra xem batch hiện tại có đầy không: Nếu $\text{len}(active\_batch) \ge Batch_{max}$, từ chối hành động cho request này.
     * **Nếu request ở trạng thái `RECOMPUTE_WAITING`**:
       * Tính toán số block prefill cơ sở: $B_{req} = \lceil p_i / B_{size} \rceil$.
       * Yêu cầu kiểm tra điều kiện nạp (batch còn chỗ, prefill của batch $< 2$, và đủ VRAM: $free\_blocks \ge B_{req}$):
         * Đưa request trở lại `active_batch` ở trạng thái prefill.
         * Khóa và trừ VRAM: $free\_blocks \leftarrow free\_blocks - B_{req}$.
         * Cấp phát $B_{req}$ blocks vật lý GPU trống cho request, lưu vào `allocated_blocks` và ghi nhận vào `block_table[request_id]`.
         * Xóa request khỏi `preempted_queue`.
     * **Nếu request ở trạng thái `SWAPPED_TO_CPU`**:
       * Tính toán số block vật lý cần để nạp lại KV Cache: $B_{restore} = \lceil (p_i + tokens\_generated) / B_{size} \rceil$.
       * Nếu GPU còn đủ bộ nhớ dự phòng ($free\_blocks \ge B_{restore}$):
         * Trừ VRAM GPU: $free\_blocks \leftarrow free\_blocks - B_{restore}$.
         * Cấp phát lại $B_{restore}$ block vật lý GPU trống cho request, cập nhật `allocated_blocks` và ghi nhận vào `block_table[request_id]`.
         * Đưa request trở lại `active_batch` với `stage = SWAP_IN_DEGRADED` và `swap_in_remaining = K_{swap} = 3`.
         * Xóa request khỏi `preempted_queue`.
         * *Chú ý*: CPU RAM của request (`request.allocated_blocks_cpu`) vẫn được giữ nguyên chưa giải phóng trên CPU RAM trong suốt 3 steps swap-in để bảo vệ an toàn dữ liệu.
       * Nếu không đủ VRAM: Giữ nguyên trạng thái tạm dừng của request.
4. **Lệnh `NoOp()`**:
   * Không thực hiện bất kỳ thay đổi cấu trúc nào đối với batch. GPU chỉ tiếp tục xử lý các công việc đang có.

### Pha 2: GPU Tiến hành Xử lý & Cập nhật Bộ nhớ (GPU Compute & Memory Allocation)
GPU tiến hành tính toán song song cho tất cả các request $k \in \text{active\_batch}$ tại step $t$:
* **Nhóm request ở `stage == PREFILL`**:
  * GPU xử lý prompt đầu vào. Không sinh thêm token đầu ra tại step này.
  * Ghi nhận thời điểm phản hồi đầu tiên: $TTFT_k = t - a_k + 1$ steps (thời gian tồn tại thực tế tính cả step nạp).
  * Chuyển đổi trạng thái của request sang giai đoạn tiếp theo: `stage = DECODE` cho step $t+1$.
* **Nhóm request ở `stage == SWAP_IN_DEGRADED`**:
  * Mô phỏng quá trình nạp lại KV Cache từ CPU RAM về GPU. Không sinh thêm token đầu ra tại step này.
  * Giảm số bước chờ: `swap_in_remaining = swap_in_remaining - 1`.
  * Nếu `swap_in_remaining == 0`:
    * Chuyển đổi trạng thái: `stage = DECODE` cho step $t+1$.
    * Giải phóng CPU RAM thực tế: $swapped\_blocks \leftarrow swapped\_blocks - request.allocated\_blocks\_cpu$ (quá trình swap-in hoàn tất an toàn).
    * Đặt lại `request.allocated_blocks_cpu = 0`.
* **Nhóm request ở `stage == DECODE`**:
  * **Cập nhật bộ nhớ KV Cache**:
    * Tổng số lượng token cần lưu trữ cho bước tiếp theo: $T_{accum} = p_k + tokens\_generated + 1$.
    * Số lượng block bộ nhớ cần thiết: $B_{needed} = \lceil T_{accum} / B_{size} \rceil$.
    * Nếu $B_{needed} > \text{len}(allocated\_blocks)$ (Cần cấp thêm 1 block mới để decode tiếp):
      * **Nếu `free_blocks > 0`**:
        * Cấp phát thêm 1 block: $free\_blocks \leftarrow free\_blocks - 1$.
        * Thêm ID block mới vào danh sách `allocated_blocks` của request và cập nhật `block_table[request_id]`.
      * **Nếu `free_blocks == 0`**:
        * **Sập bộ nhớ lập tức (OOM Crash)**: Hệ thống lập tức dừng mô phỏng (Simulation Crashed / VRAM OOM Fail) và kết thúc episode với điểm phạt OOM nặng nề. Không sinh bất kỳ token ảo nào.
    * Sau khi cấp phát bộ nhớ thành công, sinh ra chính xác 1 token đầu ra cho request: `tokens_generated = tokens_generated + 1`.

### Pha 3: Kiểm tra hoàn thành, Tự động hủy (Timeout) & Tiến trình Thời gian
1. **Kiểm tra hoàn thành**: Với mỗi request $k \in \text{active\_batch}$, hệ thống so sánh số lượng token đã sinh với độ dài response thực tế ẩn $o_k$ (độ dài thực tế từ dataset, tối đa bị giới hạn ở $o_{max} = 512$ tokens):
   * Nếu `tokens_generated == o_k` (hoặc mô hình sinh ra token kết thúc `<eos>`):
     * Ghi nhận thời gian hoàn thành và tính toán thời gian hoàn thành (Turnaround Time): $Turnaround_k = t - a_k + 1$ steps (số step thực tế request tồn tại trong hệ thống bao gồm cả step đến và step hoàn thành).
     * Giải phóng toàn bộ các block vật lý GPU của request $k$ về bể chứa: $free\_blocks \leftarrow free\_blocks + \text{len}(allocated\_blocks)$.
     * Xóa request $k$ khỏi `active_batch` và `block_table`. Đánh dấu request là thành công.
2. **Sự kiện Tự động Hủy yêu cầu của Khách hàng (Client Timeout)**:
   * Hệ thống quét qua hàng đợi `queue` và `preempted_queue` để phát hiện các request bị bỏ đói quá lâu:
   * Với mỗi request $j$ chưa hoàn thành, nếu $t - d_j > K_{timeout}$ (với $K_{timeout} = 30$ steps quá hạn):
     * Khách hàng tự động hủy yêu cầu kết nối.
     * Giải phóng bộ nhớ ảo trên CPU RAM (nếu đang ở trạng thái `SWAPPED_TO_CPU`): $swapped\_blocks \leftarrow swapped\_blocks - request.allocated\_blocks\_cpu$.
     * Xóa request $j$ khỏi hàng đợi tương ứng.
     * Áp dụng ngay lập tức **Hình phạt Hủy yêu cầu nặng** ở step này:
       $$P_{abort\_step} = P_{abort\_step} + C_{abort} \times pr_j \quad (\text{với } C_{abort} = 20.0)$$
3. **Cập nhật thời gian**: Tăng thời gian hệ thống: $t \leftarrow t + 1$.
4. **Nạp Request mới**: Kiểm tra và đưa các request mới đến tại thời điểm $t$ (sinh theo tiến trình Poisson) vào hàng đợi `queue`.
5. **Kiểm tra kết thúc Episode**:
   * Nếu `queue`, `preempted_queue` và `active_batch` đều trống rỗng (xử lý xong toàn bộ request), OR `current_time == T_max` ($5000$ steps):
     * Episode chạy mô phỏng kết thúc bình thường.

### Pha 4: Tính toán Reward và Trả về Observation (Reward & Observation)
Môi trường trả về các kết quả phản hồi cho Scheduler/Agent:
* **Observation**: Trạng thái hiện tại của `queue`, `preempted_queue`, `active_batch` và các chỉ số tài nguyên.
* **Reward**: Tính toán điểm số thưởng/phạt dựa trên các tiêu chí toán học định lượng ở Mục 3.

---

## 3. Công thức Hàm Reward (Reward Function Formulation)

Hàm Reward tại mỗi step $t$ được thiết kế với các trọng số mặc định:

$$Reward(t) = R_{throughput} - (P_{wait} + P_{SLA} + P_{abort\_step} + P_{OOM\_crash})$$

Các tham số trọng số mặc định khuyến nghị:
$$w_1 = 1.0, \quad w_2 = 0.01, \quad w_3 = 10.0, \quad C_{abort} = 20.0, \quad C_{SLA\_terminal} = 15.0, \quad C_{OOM} = -1000$$

### A. Thưởng Thông lượng (Throughput Reward)
Khuyến khích tối đa hóa số lượng token được xử lý thực tế bởi GPU ở mỗi step (cân bằng phần thưởng prefill và decode theo token thực tế được GPU xử lý):
$$R_{throughput} = w_1 \times \left( \sum_{i \in \text{prefill\_batch}} p_i + N_{active\_decode} \right)$$
*Trong đó:*
* $\sum p_i$ là tổng số prompt token được GPU chạy tính toán prefill ở step này (thành phần này được thưởng cực kỳ chuẩn xác theo khối lượng tính toán).
* $N_{active\_decode}$ là số lượng request đang hoạt động ở pha `DECODE` thực tế sinh token tại step này.

### B. Phạt Thời gian chờ (Wait Time Penalty)
$$P_{wait} = w_2 \times \sum_{i \in \text{queue}} (t - a_i)$$

### C. Phạt Vi phạm SLA (Flat/One-time SLA Penalty)
Phạt một lần duy nhất với mức phạt cố định ngay tại step đầu tiên request bị trễ deadline ($t == d_k + 1$) đối với tất cả các request chưa hoàn thành trong hệ thống, nhằm tránh bùng nổ hàm bậc hai gây sập rã học máy:
$$P_{SLA} = w_3 \times \sum_{k \in \text{active\_batch} \cup \text{queue} \cup \text{preempted\_queue}} \mathbb{I}(t == d_k + 1) \times pr_k$$
*Trong đó:*
* $\mathbb{I}(\cdot)$ là hàm chỉ thị (indicator function), trả về 1 nếu điều kiện đúng, 0 nếu sai.
* $pr_k \in \{1, 2, 3\}$ là mức độ ưu tiên của request.

### D. Phạt Khách hàng Hủy yêu cầu do chờ lâu (Abort Penalty)
$$P_{abort\_step} = \sum_{j \in \text{aborted\_requests\_in\_this\_step}} C_{abort} \times pr_j \quad (\text{với } C_{abort} = 20.0)$$

### E. Hình phạt Kết thúc Episode Đột ngột / Tồn đọng (Terminal SLA Penalty)
Khi episode kết thúc đột ngột (do VRAM OOM Crash, CPU Host OOM Crash, hoặc hết giờ $T_{max} = 5000$), toàn bộ các request chưa hoàn thành còn tồn đọng trong hệ thống sẽ bị tính phạt trừng phạt một lần duy nhất cực nặng để tránh Agent trốn phạt SLA bằng cách tự sát OOM hoặc trì hoãn:
$$P_{OOM\_crash} = \text{Nếu sập hoặc hết giờ } \sum_{u \in \text{unfinished}} C_{SLA\_terminal} \times pr_u + P_{crash} \text{ ngược lại } 0$$
*Trong đó:*
* $C_{SLA\_terminal} = 15.0$.
* $P_{crash} = C_{OOM} = -1000$ nếu sập OOM (VRAM hoặc CPU), bằng 0 nếu hoàn thành hết hoặc hết giờ $T_{max}$.
