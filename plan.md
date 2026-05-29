# Tên đề tài: Tối ưu hóa lập lịch LLM Serving
## Nhóm thực hiện
- **Đinh Việt Hoàng (23010051)**
- **Nguyễn Kim Trường (23010012)**
- **Nguyễn Quang Hiệp (23010451)**
- **Nguyễn Quốc Hiếu (23010170)**

## Giảng viên hướng dẫn

- **PSG.TS Phạm Văn Cảnh**

## 1. Mục tiêu dự án
* Xây dựng môi trường giả lập hiệu năng cao phục vụ LLM (`LLMEnv`) bằng Python phản ánh chính xác các ràng buộc vật lý thực tế (Continuous Batching, PagedAttention, Swapping).
* Thiết kế và huấn luyện bộ lập lịch thông minh dựa trên Học máy Tăng cường (Reinforcement Learning - RL) để tối ưu hóa đồng thời nhiều mục tiêu: TTFT, Turnaround Time, System Throughput và tỷ lệ đáp ứng SLA/Deadline.
* So sánh định lượng và chứng minh tính vượt trội của thuật toán lập lịch RL so với các chiến lược lập lịch truyền thống (FCFS, Priority, EDF).

---

## 2. Kế hoạch chi tiết

| Giai đoạn & Nội dung thực hiện (What to do) | Kết quả mong đợi (Expected results) |
| :--- | :--- |
| **Giai đoạn 1: Chuẩn bị dữ liệu thực nghiệm** | |
| Tải và xử lý một tập con đại diện của dataset `LMSYS-Chat-1M`. | Tệp dữ liệu sạch chứa các cặp độ dài `(prompt_len, response_len)` thực tế. |
| Thống kê và mô hình hóa phân phối xác suất độ dài của Prompt và Response. | Các tham số phân phối (hoặc phân phối thực nghiệm) dùng để sinh request ngẫu nhiên trong giả lập. |
| **Giai đoạn 2: Xây dựng Môi trường Giả lập (`LLMEnv`)** | |
| Triển khai lớp giả lập `LLMEnv` (Python/Gymnasium) quản lý trạng thái GPU VRAM và CPU RAM theo Block. | Lớp `LLMEnv` thực thi chính xác 4 pha chuyển trạng thái ở mỗi step, hỗ trợ các cơ chế nạp, dừng (swap/recompute), swap-in delay và client timeout. |
| Tích hợp tiến trình Poisson để tự động sinh dòng request online theo các kịch bản tải khác nhau. | Môi trường có khả năng tạo dòng request ngẫu nhiên liên tục theo thời gian thực mô phỏng thực tế. |
| Triển khai các bộ đếm đo lường hiệu năng và tính toán hàm Reward tổng hợp tại mỗi step. | Hệ thống telemetry trả về chính xác TTFT, Turnaround, Throughput, SLA violation và điểm số Reward. |
| **Giai đoạn 3: Phát triển các thuật toán Lập lịch Baselines** | |
| Lập trình bộ lập lịch FCFS (First-Come, First-Served). | Baseline 1 hoạt động ổn định làm mốc so sánh cơ sở. |
| Lập trình bộ lập lịch EDF (Earliest Deadline First). | Baseline 2 ưu tiên thời hạn làm mốc so sánh về mặt SLA. |
| Lập trình bộ lập lịch dựa trên độ ưu tiên tĩnh (Priority-based). | Baseline 3 ưu tiên xếp hạng khách hàng. |
| **Giai đoạn 4: Thiết kế và Huấn luyện Agent RL** | |
| Định nghĩa không gian quan sát (Observation Space) và không gian hành động (Action Space) dạng danh sách. | Thiết kế tensor đầu vào/đầu ra chuẩn hóa cho mạng thần kinh. |
| Triển khai và huấn luyện mô hình RL (ví dụ: DQN hoặc PPO sử dụng Stable-Baselines3). | Agent học được chính sách lập lịch phòng thủ thông minh, biết cách tự động preempt/resume để bảo vệ GPU và SLA. |
| Tối ưu hóa siêu tham số (Hyperparameter Tuning) và tinh chỉnh hàm Reward. | Mô hình hội tụ ổn định, đạt điểm reward trung bình tăng dần qua các epoch. |
| **Giai đoạn 5: Đánh giá & So sánh** | |
| Chạy thử nghiệm các thuật toán trên cùng một kịch bản dòng request thử thách (Poisson tải cao). | Bảng số liệu định lượng chi tiết so sánh các metrics giữa FCFS, EDF, Priority và Agent RL. |
| Trực quan hóa kết quả bằng đồ thị (đường cong học tập, biểu đồ phân bố TTFT/Turnaround). | Các file ảnh đồ thị trực quan chứng minh trực tiếp hiệu quả của mô hình RL. |

---

## 3. Sản phẩm bàn giao cuối cùng (Final Deliverables)
1. **Hệ thống Giả lập Phục vụ LLM (LLM Serving Simulator)**: Môi trường mô phỏng giúp kiểm thử, dự báo hiệu năng và chi phí vận hành của hệ thống GPU dưới các kịch bản tải thực tế ngoài đời thực mà không cần đầu tư phần cứng GPU đắt đỏ ngay từ đầu.
2. **Bộ Não Điều phối AI Thông minh (AI-Powered Scheduling Engine)**: Giải pháp lập lịch tự động tối ưu hóa việc phân bổ bộ nhớ đệm KV Cache trên GPU/CPU, cắt giảm tối đa thời gian phản hồi đầu tiên (TTFT) cho người dùng, và bảo vệ hệ thống an toàn trước nguy cơ sập nguồn OOM.
3. **Bộ Công cụ So sánh và Đánh giá SLA (SLA Optimization & Benchmarking Suite)**: Báo cáo định lượng trực quan giúp người quản trị hệ thống có đầy đủ dữ liệu để đưa ra các quyết định cấu hình gói dịch vụ (VIP/Regular) và cam kết thời hạn (SLA) tối ưu nhất cho khách hàng.
