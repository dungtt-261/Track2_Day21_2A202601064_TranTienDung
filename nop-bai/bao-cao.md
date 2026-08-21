# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Trần Tiến Dũng |
| MSSV | 2A202601064 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/dungtt-261/Track2_Day21_2A202601064_TranTienDung |
| Ngày nộp | 21/08/2026 |

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 200 | 0.05 | 5 | 0.7037 | 0.8720 |
| 5 | 300 | 0.1 | 4 | 0.7123 | 0.8740 |

**Đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5` — f1 cao nhất.

**Lý do:** Lần có accuracy cao nhất là lần 1 (0.8780) chứ không phải lần 3; chọn theo
accuracy tôi đã lấy nhầm mô hình bắt được ít ca thu nhập cao hơn. Hạ `learning_rate` còn
0.05 mà giữ `n_estimators` (lần 4 so với lần 3) làm f1 giảm, vì mỗi cây đóng góp ít hơn
nhưng số vòng boosting không tăng để bù. Lần 2 minh họa rõ nhất rủi ro của accuracy: f1 rơi
xuống 0.6051 và bị chặn, trong khi accuracy vẫn 0.8460, trông rất khả quan.

## 2. Vì Sao Ngưỡng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult chỉ có 24,8% mẫu thuộc lớp thu nhập trên 50K, tức mất cân bằng 75/25. Một mô hình
vô dụng luôn trả lời "thu nhập thấp" vẫn đạt accuracy 0,752 mà không bắt được ca thu nhập
cao nào — con số đó chỉ phản ánh việc đoán đúng lớp đa số. F1 của lớp dương là trung bình
điều hòa của precision và recall trên riêng lớp thu nhập cao, nên phạt cả việc bỏ sót lẫn
báo động nhầm. Cũng vì vậy không dùng `average="weighted"` hay `"macro"`: cả hai trộn điểm
của lớp đa số vào, kéo giá trị lên và làm ngưỡng 0.65 mất ý nghĩa.

## 3. Khó Khăn và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `import mlflow` lỗi thiếu `pkg_resources` | setuptools ≥ 81 đã gỡ module này | Ghim `setuptools<81` |
| `dvc push` báo AccessDenied | IAM group không có quyền S3 | Tạo policy scope vào `income-lab-*` |
| Push không kích hoạt pipeline | Fork mặc định tắt Actions cho push | Bật thủ công ở tab Actions |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm f1 tăng 0,0205 — cải thiện có thật nhưng khiêm tốn,
phù hợp với việc hai nửa dữ liệu cùng phân phối nên không mang thông tin mới. Thứ được kiểm
chứng ở Bước 3 là quy trình tự động: một commit `.dvc` kéo trọn vòng huấn luyện, kiểm tra
chất lượng và triển khai lại. `f1` do CI tính trùng đến từng chữ số với máy cá nhân.

## 5. Phần Bonus Đã Thực Hiện

- [ ] **Bonus 1** — DagsHub: chưa làm.
- [x] **Bonus 2** — Quét ngưỡng 0.1–0.9: tối ưu tại **0.30** cho f1 **0.7537**, hơn ngưỡng
  mặc định 0.5 là 0,0183.
- [x] **Bonus 3** — `outputs/detail.txt` có confusion matrix và precision/recall từng lớp.
  Lớp thu nhập cao: precision 0.828, recall 0.661 — bỏ sót 42/124 ca. Bỏ sót tốn kém hơn
  gán nhầm, vì người đó mất cơ hội tiếp cận, còn gán nhầm chỉ tốn một bước xác minh.
- [x] **Bonus 4** — So sánh f1 mới với `artifacts/current/report.json` trước khi release.
  Việc này lộ ra lỗ hổng: upload nằm trong job Train nên chạy **trước** Quality Gate, model
  hỏng vẫn ghi đè `current`. Đã sửa thành upload vào `candidate/` rồi mới thăng cấp sau khi
  qua cả hai cổng. Thử với bộ f1 0.7014 (qua ngưỡng nhưng kém bản đang chạy 0.7354):
  pipeline dừng đúng chỗ, `current/` giữ nguyên model tốt.
- [x] **Bonus 5** — Cảnh báo khi tỷ lệ lớp dương lệch quá 5 điểm phần trăm so với 24,8%.
