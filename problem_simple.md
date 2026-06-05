# Đề tài Đơn giản hóa: Tối ưu hóa lập lịch yêu cầu cho LLM Serving (Token-based Memory)

---

## 1. Thực tế hệ thống đơn giản hóa (The Reality)
Môi trường giả lập mô phỏng các cơ chế của một GPU Worker Instance phục vụ LLM ở cấp độ token trực tiếp (không quản lý phân bổ block vật lý):

### A. Cơ chế Continuous Batching & Năng lực GPU
* Hệ thống hoạt động theo từng chu kỳ sinh token (**step / iteration**).
* Mỗi request khi chạy trải qua hai pha:
  * **Prefill Phase (Xử lý đầu vào)**: Tính toán prompt đầu vào ($p_i$) của request mới nạp. Diễn ra trong đúng 1 step đầu tiên và là nguyên tử (không thể preempt giữa chừng).
  * **Decode Phase (Sinh đầu ra)**: Mỗi step sinh ra đúng 1 token cho request.
* **Giới hạn GPU**: 
  * Kích thước batch tối đa: $Batch_{max} = 16$.
  * Giới hạn prefill song song: GPU chỉ xử lý tối đa **2 requests chạy Prefill đồng thời** trong 1 step.

### B. Giới hạn bộ nhớ VRAM GPU (Token-based)
* Không dùng PagedAttention (không chia block). Dung lượng bộ nhớ được đo trực tiếp bằng số lượng token tối đa có thể chứa.
* **Dung lượng GPU VRAM tối đa**: $T_{gpu\_max} = 16000$ tokens.
* Bộ nhớ chiếm dụng bởi một request $i$ đang hoạt động tại step $t$: $T_i = p_i + g_i$ (với $p_i$ là độ dài prompt, $g_i$ là số token response đã sinh).
* Bộ nhớ GPU đang sử dụng: $T_{gpu\_used} = \sum_{k \in \mathcal{B}_{active}} (p_k + g_k)$.
* Nếu tại bất kỳ bước decode nào, việc sinh token làm $T_{gpu\_used} + 1 > T_{gpu\_max}$ -> Xảy ra quá tải VRAM (**VRAM OOM**): Request bị hoãn sinh token ở step này (giữ nguyên $g_k$), không tăng bộ nhớ, hệ thống chịu một khoản phạt OOM mà không bị dừng mô phỏng.
* Giới hạn response thực tế: Cắt tối đa tại $o_{max} = 512$ tokens.

### C. Cơ chế Swapping (Bộ nhớ ảo trên CPU RAM) & Độ trễ
* Khi preempt một request bằng chiến lược Swap, toàn bộ số token hiện tại của nó ($p_i + g_i$) được chuyển sang CPU RAM.
* **Dung lượng CPU RAM tối đa**: $T_{cpu\_max} = 32000$ tokens. Nếu hành động Swap vượt quá giới hạn này -> Bị từ chối (thất bại, request giữ nguyên trong batch) và hệ thống chịu điểm phạt tràn CPU RAM.
* Khi Resume một request từ CPU RAM, request phải chịu **độ trễ swap-in là $K_{swap} = 3$ steps**. 
* Trong thời gian swap-in, request chiếm dụng bộ nhớ VRAM GPU nhưng không sinh token. Bộ nhớ trên CPU RAM chỉ giải phóng sau khi hoàn tất swap-in.

### D. Hàng đợi & Client Timeout
* Dòng request online đến theo tiến trình Poisson với các thuộc tính: thời điểm đến $a_i$, độ dài prompt $p_i$, độ ưu tiên $pr_i \in \{1, 2, 3\}$, thời hạn hoàn thành $d_i$.
* **Client Timeout**: Nếu một request nằm chờ quá lâu trong hàng đợi và bị trễ deadline vượt quá $K_{timeout} = 30$ steps, khách hàng tự động hủy kết nối (Abort). Hệ thống giải phóng bộ nhớ của request đó và chịu phạt hủy yêu cầu rất nặng.

---

## 2. Bài toán lập lịch (The Problem to Solve)
Tối ưu hóa chính sách lập lịch để:
1. Giảm thiểu thời gian phản hồi đầu tiên (TTFT).
2. Giảm thiểu thời gian hoàn thành (Turnaround Time).
3. Đảm bảo tỷ lệ đáp ứng SLA/Deadline (ưu tiên priority cao).
4. Tối đa hóa thông lượng (Throughput).
5. Tránh lỗi OOM trên cả GPU và CPU.

---

## 3. Không gian hành động của Bộ lập lịch (Action Space)
Tại đầu mỗi step $t$, Scheduler chọn một hành động:
1. **`Promote(queue_indices: List[int])`**: Nạp tối đa 2 requests mới từ hàng đợi `queue` vào batch. Yêu cầu GPU còn đủ chỗ trống trong batch và đủ VRAM để khởi tạo prompt: $T_{gpu\_max} - T_{gpu\_used} \ge \sum p_i$.
2. **`Preempt(active_batch_indices: List[int], strategy: str)`**: Trục xuất các request được chọn ra khỏi batch để giải phóng VRAM.
   * *Preempt-by-Recompute*: Xóa sạch token đã sinh, đưa về hàng đợi (phát sinh phạt bằng số lượng token đã sinh bị mất). Giải phóng VRAM GPU hoàn toàn.
   * *Preempt-by-Swap*: Chuyển KV Cache ($p_i + g_i$ tokens) sang CPU RAM nếu đủ dung lượng.
3. **`Resume(preempted_queue_indices: List[int])`**: Khôi phục xử lý các request bị tạm dừng từ `preempted_queue`.
   * *Recompute*: Yêu cầu đủ VRAM chạy lại prefill.
   * *Swap*: Yêu cầu đủ VRAM để nạp lại cache. Chờ 3 steps swap-in.
4. **`NoOp()`**: Giữ nguyên trạng thái batch.
