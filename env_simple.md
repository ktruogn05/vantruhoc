# Đặc tả Môi trường Giả lập Đơn giản hóa (`LLMEnvSimple`)

---

## 1. Trạng thái Môi trường (State Representation)

Tại mỗi step $t$, trạng thái mô phỏng bao gồm:

### A. Quản lý Bộ nhớ (Token-based Memory Pool)
* $T_{\text{gpu\_max}} = 16000$: Số lượng token tối đa chứa được trên GPU VRAM.
* $T_{\text{cpu\_max}} = 32000$: Số lượng token tối đa chứa được trên CPU RAM.
* `gpu_tokens_used`: Tổng số token đang chiếm dụng trên GPU VRAM.
* `cpu_tokens_used`: Tổng số token đang chiếm dụng trên CPU RAM.

### B. Danh sách Batch hoạt động (`active_batch`)
Tập hợp các request $\mathcal{B}_{\text{active}}$ đang được GPU xử lý ($|\mathcal{B}_{\text{active}}| \le 16$).
Với mỗi request $i \in \mathcal{B}_{\text{active}}$, lưu trữ:
* `stage`: Trạng thái xử lý (`PREFILL`, `SWAP_IN_DEGRADED`, `DECODE`).
* `swap_in_remaining` ($\tau_{\text{remain}}$): Số bước chờ swap-in (chỉ dùng khi `stage == SWAP_IN_DEGRADED`).
* `prompt_tokens` ($p_i$): Độ dài prompt.
* `tokens_generated` ($g_i$): Số token response đã sinh.
* `priority` ($pr_i$): Độ ưu tiên ($1, 2, 3$).
* `arrival_time` ($a_i$): Thời điểm đến.
* `deadline` ($d_i$): Thời điểm quá hạn.
* `cpu_tokens_held`: Số token đang giữ trên CPU RAM (bằng $p_i + g_i$ khi đang trong pha swap-in, mặc định = 0).

### C. Các Hàng đợi (Queues)
* `queue`: Request mới chưa từng được nạp, sắp xếp theo thời điểm đến.
* `preempted_queue`: Request đang chạy dở bị tạm dừng. Nhãn lưu trữ:
  * `RECOMPUTE_WAITING`: Giải phóng hoàn toàn bộ nhớ, cần chạy lại prefill từ đầu khi resume.
  * `SWAPPED_TO_CPU`: Lưu giữ KV Cache ($p_i + g_i$ tokens) trên CPU RAM.

### D. Trục thời gian
* `current_time` ($t$): Tăng $+1$ sau mỗi step.

---

## 2. Quy trình Chuyển trạng thái (`env.step(action)`)

### Pha 1: Thực thi Hành động Lập lịch (Scheduler Actions)

Khởi tạo các thông số phạt tạm thời cho step $t$: $P_{\text{abort}} = 0$, $P_{\text{recompute}} = 0$, $P_{\text{oom}} = 0$, $P_{\text{cpu\_overflow}} = 0$.

Hệ thống xử lý một lệnh duy nhất từ Scheduler tại đầu step $t$:

1. **`Promote(queue_indices: List[int])`** (tối đa 2 requests):
   * Duyệt qua từng chỉ mục $idx$ được chọn trong `queue`:
     * Lấy request $i$ tại chỉ mục đó.
     * Kiểm tra điều kiện nạp: Batch chưa đầy ($|\mathcal{B}_{\text{active}}| < 16$), tổng prefill song song ở step này không vượt quá 2, và GPU đủ bộ nhớ trống: $T_{\text{gpu\_max}} - \text{gpu\_tokens\_used} \ge p_i$.
     * Đưa request vào `active_batch` với `stage = PREFILL`, $g_i = 0$.
     * Cập nhật bộ nhớ GPU: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} + p_i$.
     * Xóa request khỏi `queue`.
     * Nếu không đủ điều kiện: giữ nguyên ở `queue`.

2. **`Preempt(active_batch_indices: List[int], strategy: str)`**:
   * Với từng chỉ mục $idx$ trong `active_batch` (tương ứng request $i$):
     * Từ chối preempt nếu request đang ở `stage == PREFILL` (bảo vệ tính nguyên tử của pha prefill).
     * Xóa request khỏi `active_batch`.
     * **Nếu request ở `stage == SWAP_IN_DEGRADED`**:
       * Thu hồi VRAM GPU: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} - (p_i + g_i)$.
       * Nếu `strategy == Recompute`: 
         * Giải phóng CPU RAM: $\text{cpu\_tokens\_used} \leftarrow \text{cpu\_tokens\_used} - \text{cpu\_tokens\_held}_i$.
         * Đưa request vào `preempted_queue` nhãn `RECOMPUTE_WAITING`.
         * Cộng dồn phạt recompute của step: $P_{\text{recompute}} \leftarrow P_{\text{recompute}} + 1.0 \times g_i$.
       * Nếu `strategy == Swap`: giữ nguyên CPU RAM, đưa request vào `preempted_queue` nhãn `SWAPPED_TO_CPU`.
     * **Nếu request ở `stage == DECODE`**:
       * **Chiến lược `Recompute`**:
         * Thu hồi VRAM GPU: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} - (p_i + g_i)$.
         * Cộng dồn phạt recompute của step: $P_{\text{recompute}} \leftarrow P_{\text{recompute}} + 1.0 \times g_i$.
         * Reset token đã sinh: $g_i = 0$.
         * Đưa request vào `preempted_queue` nhãn `RECOMPUTE_WAITING`.
       * **Chiến lược `Swap`**:
         * Kiểm tra CPU RAM: Nếu $\text{cpu\_tokens\_used} + (p_i + g_i) > T_{\text{cpu\_max}}$:
           * Không thực hiện Swap (hành động thất bại, request giữ nguyên trong `active_batch` ở `stage == DECODE`).
           * Cộng dồn phạt tràn CPU RAM của step: $P_{\text{cpu\_overflow}} \leftarrow P_{\text{cpu\_overflow}} + 50.0$.
         * Nếu đủ CPU RAM:
           * Thu hồi GPU VRAM: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} - (p_i + g_i)$.
           * Đưa sang CPU RAM: $\text{cpu\_tokens\_used} \leftarrow \text{cpu\_tokens\_used} + (p_i + g_i)$, lưu trữ $\text{cpu\_tokens\_held}_i = p_i + g_i$.
           * Đưa request vào `preempted_queue` nhãn `SWAPPED_TO_CPU`.

3. **`Resume(preempted_queue_indices: List[int])`** (tối đa 2 requests):
   * Duyệt qua từng chỉ mục $idx$ được chọn trong `preempted_queue` (tương ứng request $i$):
     * Kiểm tra batch chưa đầy ($|\mathcal{B}_{\text{active}}| < 16$).
     * **Nếu trạng thái `RECOMPUTE_WAITING`**:
       * Kiểm tra GPU đủ bộ nhớ khởi chạy prefill ($T_{\text{gpu\_max}} - \text{gpu\_tokens\_used} \ge p_i$), prefill song song ở step này $< 2$.
       * Đưa request vào `active_batch` với `stage = PREFILL`, $g_i = 0$.
       * Trừ GPU VRAM: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} + p_i$.
       * Xóa khỏi `preempted_queue`.
     * **Nếu trạng thái `SWAPPED_TO_CPU`**:
       * Kiểm tra GPU đủ bộ nhớ chứa KV Cache cũ ($T_{\text{gpu\_max}} - \text{gpu\_tokens\_used} \ge p_i + g_i$).
       * Đưa request vào `active_batch` với `stage = SWAP_IN_DEGRADED` và $\tau_{\text{remain}} = 3$.
       * Trừ GPU VRAM: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} + (p_i + g_i)$.
       * Xóa khỏi `preempted_queue`. (Giữ bộ nhớ CPU RAM của request chưa giải phóng cho đến khi hoàn thành swap-in).

4. **`NoOp()`**:
   * Không thay đổi cấu trúc batch.

---

### Pha 2: GPU Xử lý & Cập nhật Bộ nhớ

Xử lý tính toán cho các request $k \in \mathcal{B}_{\text{active}}$:

*   **Nếu `stage == PREFILL`**:
    *   Xử lý prompt đầu vào. Không sinh token mới ở step này.
    *   Ghi nhận $\text{TTFT}_k = t - a_k + 1$.
    *   Chuyển trạng thái: `stage = DECODE` cho step $t+1$.
*   **Nếu `stage == SWAP_IN_DEGRADED`**:
    *   Giảm thời gian chờ: $\tau_{\text{remain}, k} = \tau_{\text{remain}, k} - 1$.
    *   Nếu $\tau_{\text{remain}, k} == 0$:
        *   Chuyển trạng thái: `stage = DECODE` cho step $t+1$.
        *   Giải phóng bộ nhớ CPU RAM thực tế: $\text{cpu\_tokens\_used} \leftarrow \text{cpu\_tokens\_used} - \text{cpu\_tokens\_held}_k$.
        *   Đặt $\text{cpu\_tokens\_held}_k = 0$.
*   **Nếu `stage == DECODE`**:
    *   Kiểm tra bộ nhớ GPU nếu sinh thêm 1 token:
        *   Nếu $\text{gpu\_tokens\_used} + 1 > T_{\text{gpu\_max}}$:
            *   Áp dụng phạt VRAM OOM cho step này: $P_{\text{oom}} \leftarrow P_{\text{oom}} + 10.0$.
            *   Không tăng `gpu_tokens_used`, không sinh thêm token ở step này (giữ nguyên $g_k$, vẫn giữ `stage = DECODE` cho step $t+1$).
        *   Nếu đủ bộ nhớ GPU:
            *   Yêu cầu bộ nhớ: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} + 1$.
            *   Sinh thành công 1 token: $g_k = g_k + 1$.

---

### Pha 3: Kiểm tra hoàn thành, Client Timeout & Cập nhật thời gian

1.  **Kiểm tra hoàn thành**:
    *   Nếu $g_k == o_k$ (hoặc sinh xong):
        *   Tính $\text{Turnaround}_k = t - a_k + 1$.
        *   Giải phóng bộ nhớ GPU: $\text{gpu\_tokens\_used} \leftarrow \text{gpu\_tokens\_used} - (p_k + g_k)$.
        *   Xóa khỏi `active_batch`.
2.  **Khách hàng tự hủy yêu cầu (Client Timeout)**:
    *   Quét `queue` và `preempted_queue`.
    *   Nếu $t - d_j > 30$:
        *   Xóa request khỏi hàng đợi tương ứng.
        *   Nếu ở dạng `SWAPPED_TO_CPU`: Giải phóng CPU RAM: $\text{cpu\_tokens\_used} \leftarrow \text{cpu\_tokens\_used} - \text{cpu\_tokens\_held}_j$.
        *   Áp dụng phạt hủy yêu cầu của step hiện tại: $P_{\text{abort}} = P_{\text{abort}} + 20.0 \times pr_j$.
3.  **Cập nhật thời gian**: $t \leftarrow t + 1$.
4.  **Nạp Request mới** đến theo tiến trình Poisson tại thời điểm $t$ vào `queue`.
5.  **Kiểm tra kết thúc Episode**:
    *   Kết thúc nếu `queue`, `preempted_queue`, và `active_batch` đều trống, HOẶC đạt $t == 5000$.

---

### Pha 4: Tính toán Reward & Trả về Observation

*   **Observation**: Danh sách trạng thái các hàng đợi, batch hoạt động, và dung lượng bộ nhớ trống.
*   **Reward**:
    $$\text{Reward}(t) = R_{\text{throughput}} - (P_{\text{wait}} + P_{\text{SLA}} + P_{\text{abort}} + P_{\text{recompute}} + P_{\text{oom}} + P_{\text{cpu\_overflow}} + P_{\text{crash}})$$
    *   $R_{\text{throughput}} = 1.0 \times N_{\text{decode}}$ (chỉ thưởng cho token sinh thành công ở step hiện tại để tránh loop reward prefill).
    *   $P_{\text{wait}} = 0.01 \times \sum_{i \in \text{queue} \cup \text{preempted\_queue}} (t - a_i)$ (phạt chờ cho mọi request đang xếp hàng hoặc bị trì hoãn).
    *   $P_{\text{SLA}} = 10.0 \times \sum_{k \in \text{all}} \mathbb{I}(t = d_k + 1) \times pr_k$.
    *   $P_{\text{abort}}$ là tổng phạt từ các request bị hủy ở step $t$.
    *   $P_{\text{recompute}}$ là tổng phạt bằng lượng token đã sinh bị mất do preempt bằng Recompute ở step $t$.
    *   $P_{\text{oom}}$ là phạt bộ nhớ VRAM OOM ở step hiện tại.
    *   $P_{\text{cpu\_overflow}}$ là phạt vượt quá bộ nhớ CPU RAM khi preempt ở step hiện tại.
    *   $P_{\text{crash}} = 15.0 \times \sum_{u \in \text{unfinished}} pr_u$ (phạt cho các request chưa hoàn thành nếu đạt giới hạn thời gian $t == 5000$).
