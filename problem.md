# Đề tài: Tối ưu hóa lập lịch yêu cầu cho hệ thống phục vụ mô hình ngôn ngữ lớn (LLM Serving)

---

## 1. Thực tế hệ thống (The Reality)
Môi trường giả lập mô phỏng các cơ chế vật lý và kỹ thuật thực tế của một thực thể phục vụ LLM (**GPU Worker Instance**):

### A. Cơ chế Continuous Batching & Giới hạn Năng lực GPU
* Hệ thống hoạt động theo từng chu kỳ sinh token (**step / iteration**).
* Mỗi request khi chạy sẽ trải qua hai pha xử lý:
  * **Prefill Phase (Xử lý đầu vào)**: GPU tính toán toàn bộ prompt đầu vào ($p_i$) của request mới nạp. Pha này tốn nhiều năng lực tính toán nhưng chỉ diễn ra trong đúng 1 step đầu tiên và là nguyên tử (atomic - không thể bị ngắt quãng/preempt giữa chừng).
  * **Decode Phase (Sinh đầu ra)**: GPU sinh tuần tự từng token một cho request. Mỗi step sinh ra đúng 1 token. Pha này tốn ít năng lực tính toán nhưng chiếm dụng bộ nhớ tăng dần theo thời gian.
* Bộ lập lịch (Scheduler) có thể nạp request mới từ hàng đợi vào batch hoặc tạm dừng/trục xuất request đang chạy tại đầu mỗi step mà không cần chờ toàn bộ batch kết thúc.
* **Giới hạn GPU**: 
  * Kích thước batch tối đa: $Batch_{max} = 16$.
  * Giới hạn năng lực xử lý prefill: GPU chỉ có thể xử lý tối đa **2 requests chạy Prefill đồng thời** trong cùng 1 step.

### B. Cơ chế PagedAttention & Giới hạn bộ nhớ VRAM GPU
* Bộ nhớ VRAM của GPU dành cho KV Cache được chia thành một tập hợp gồm $N_{blocks} = 1000$ khối vật lý (Physical Blocks) cố định. Mỗi block chứa tối đa $B_{size} = 16$ tokens.
* Mỗi request khi đang hoạt động trong batch đòi hỏi được cấp phát thêm block mới từ bể chứa (block pool) khi số lượng token tích lũy của nó (Prompt + số lượng token Response đã sinh) vượt quá dung lượng chứa của các block hiện có.
* Nếu GPU hết block vật lý trống tại bất kỳ bước decode nào cần cấp phát thêm bộ nhớ, hệ thống sẽ **sập nguồn ngay lập tức do lỗi cạn kiệt bộ nhớ (OOM Crash)** và kết thúc lượt chạy (episode) với hình phạt cực nặng.
* Để đảm bảo an toàn bộ nhớ GPU, tất cả các response thực tế lấy từ dataset sẽ được cắt tối đa tại $o_{max} = 512$ tokens.

### C. Cơ chế Swapping (Bộ nhớ ảo trên CPU RAM) & Độ trễ khôi phục
* Khi một request bị tạm dừng bằng chiến lược Swap, bộ nhớ KV Cache của nó được chuyển từ VRAM GPU sang RAM của máy chủ (CPU Host RAM).
* Bộ nhớ CPU RAM có giới hạn tối đa là $N_{cpu\_blocks} = 2000$ blocks. Nếu số lượng block bị swap vượt quá giới hạn này, hệ thống sẽ gặp lỗi **Host OOM Crash** và sập mô phỏng.
* Việc chuyển dữ liệu giữa GPU và CPU tốn băng thông và độ trễ. Khi khôi phục (Resume) một request từ CPU RAM, request đó phải trải qua **độ trễ swap-in là $K_{swap} = 3$ steps**. Trong thời gian swap-in, request chiếm dụng bộ nhớ VRAM nhưng GPU không thể thực hiện sinh token cho nó. Bộ nhớ trên CPU RAM chỉ thực sự được giải phóng sau khi hoàn tất $K_{swap}$ steps swap-in.

### D. Hàng đợi trực tuyến & Cơ chế Tự động Hủy Yêu cầu của Khách hàng (Client Timeout)
* Các yêu cầu mới $R_i$ đến hệ thống trực tuyến theo thời gian thực (Online). Mỗi yêu cầu $i$ được biết trước các thuộc tính lúc đến:
  * $a_i$ (Arrival Time): Thời điểm yêu cầu đến hệ thống.
  * $p_i$ (Prompt Length): Số lượng token đầu vào.
  * $pr_i$ (Priority): Mức độ ưu tiên của yêu cầu ($1, 2, 3$ tương ứng thấp, trung bình, cao).
  * $d_i$ (Deadline): Thời điểm tối đa yêu cầu phải được hoàn thành ($d_i = a_i + \text{SLA}_i$).
* **Sự kiện Khách hàng tự hủy yêu cầu (Client Timeout)**:
  * Vì Bộ lập lịch (Scheduler) không có quyền chủ động hủy bỏ yêu cầu của khách hàng, môi trường giả lập tự động áp dụng cơ chế Client Timeout thực tế:
  * Nếu một request nằm chờ quá lâu trong hàng đợi (`queue` hoặc `preempted_queue`) và bị trễ deadline vượt quá $K_{timeout} = 30$ steps, **khách hàng (Môi trường) sẽ tự động hủy kết nối (Abort)**.
  * Khi sự kiện này xảy ra, môi trường sẽ tự động giải phóng toàn bộ bộ nhớ của request đó (GPU VRAM hoặc CPU RAM), xóa request khỏi hệ thống và áp dụng hình phạt hủy yêu cầu rất nặng lên hệ thống để cảnh cáo chính sách lập lịch tồi làm mất khách hàng.

---

## 2. Bài toán cần giải quyết (The Problem to Solve)
Mục tiêu là tìm kiếm một **chính sách lập lịch (Scheduling Policy)** nhằm tối ưu hóa hàm mục tiêu đa chỉ số sau:
1. **Cực tiểu hóa thời gian phản hồi đầu tiên (Time to First Token - TTFT)**: Giảm thiểu thời gian chờ đợi của request từ lúc đến ($a_i$) cho đến khi bắt đầu sinh token đầu tiên (hoàn thành pha Prefill).
2. **Cực tiểu hóa thời gian hoàn thành yêu cầu (Turnaround Time)**: Tổng thời gian từ lúc request đến cho đến khi nhận được toàn bộ kết quả.
3. **Cực tiểu hóa tỷ lệ vi phạm SLA / Trễ Deadline**: Đặc biệt ưu tiên bảo vệ các request có độ ưu tiên cao ($pr_i$ lớn) không bị hoàn thành sau thời điểm deadline $d_i$.
4. **Cực đại hóa thông lượng hệ thống (System Throughput)**: Tổng số token (cả prompt và response) được GPU xử lý thực tế trên một đơn vị thời gian.
5. **Tránh sập hệ thống (OOM)**: Luôn đảm bảo không xảy ra sập nguồn GPU VRAM OOM hoặc CPU RAM Host OOM.

**Điều kiện vận hành & Episode**:
* Tại thời điểm $t=0$, hệ thống bắt đầu với `current_time = 0`, hàng đợi trống và GPU trống. Các request được sinh ngẫu nhiên từ $t > 0$ theo tiến trình Poisson.
* Episode kết thúc khi: 
  1. Đã xử lý xong toàn bộ các request trong kịch bản (cả queue, preempted_queue và batch đều trống).
  2. Đạt giới hạn thời gian tối đa $T_{max} = 5000$ steps.
  3. Xảy ra OOM Crash (sập VRAM GPU hoặc CPU Host RAM).

---

## 3. Không gian hành động của Bộ lập lịch (The Action Space)
Tại đầu mỗi step $t$, dựa trên trạng thái quan sát, Bộ lập lịch/Agent đưa ra **một quyết định lập lịch duy nhất** từ tập hành động hợp lệ sau:

1. **`Promote(request_ids: List[str])`**:
   * Di chuyển **một hoặc tối đa 2** requests cùng lúc từ hàng đợi ngoài vào active batch để bắt đầu xử lý (khai thác tối đa năng lực prefill song song của GPU).
   * *Điều kiện*: Batch sau khi nạp không vượt quá $Batch_{max}$, số prefill được thêm vào không làm tổng số prefill ở step đó vượt quá 2, và GPU phải còn đủ block trống để khởi tạo prompt: $free\_blocks \ge \sum \lceil p_i / B_{size} \rceil$.
2. **`Preempt(request_ids: List[str], strategy: str)`**:
   * Trục xuất **một hoặc nhiều** request đang chạy ra khỏi active batch cùng lúc để giải phóng VRAM GPU ngay lập tức (cứu hệ thống khỏi OOM). Các chiến lược:
     * **Preempt-by-Recompute**: Xóa sạch KV Cache của các request đó trên GPU.
     * **Preempt-by-Swap**: Chuyển KV Cache của các request sang CPU RAM.
3. **`Resume(request_ids: List[str])`**:
   * Khôi phục xử lý cho **một hoặc nhiều** request bị tạm dừng cùng lúc.
     * Đối với *Recompute*: Yêu cầu VRAM GPU thỏa mãn điều kiện cấp phát prompt, đưa request vào active batch ở trạng thái prefill.
     * Đối với *Swap*: Yêu cầu VRAM GPU thỏa mãn điều kiện cấp phát lại toàn bộ KV cache cũ. Đưa request vào trạng thái swap-in trong $K_{swap}$ steps trước khi tiếp tục decode.
4. **`NoOp()`**:
   * Không thay đổi cấu trúc batch hiện tại. Hệ thống tự động tiến hành pha xử lý tính toán và sinh token của GPU cho các request đang có.

---

## 4. Dữ liệu thực nghiệm & Giả lập
* **Bộ dữ liệu:** Sử dụng tập con của **LMSYS-Chat-1M** để trích xuất thống kê phân phối thực tế của $p_i$ và $o_i$ phục vụ việc sinh các kịch bản request ngẫu nhiên trong môi trường giả lập.
* **Mô phỏng dòng yêu cầu:** Sử dụng tiến trình Poisson để mô phỏng thời điểm yêu cầu đến ($a_i$) nhằm thử thách bộ lập lịch dưới các mức độ tải hệ thống khác nhau.
