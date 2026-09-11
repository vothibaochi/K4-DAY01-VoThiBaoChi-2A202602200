# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không có thay đổi, chỉ chạy đúng theo luồng 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): biểu thị lớp được model xếp hạng cho toàn bộ ảnh chứ không phải đối tượng riêng lẻ
- Record này mô tả toàn ảnh như thế nào?: Nếu nhìn toàn bộ ảnh và chỉ cần chọn một nhãn phù hợp nhất cho bức ảnh này, lớp nào có xác suất cao nhất?” Vì vậy, `rank=1` là lớp nhiều khả năng nhất ở cấp độ ảnh, chứ không phải nhãn của vật thể nào đó trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?: Class list đến từ taxonomy đi kèm checkpoint, ở đây là `ImageNet-1K` cho mô hình phân loại ảnh. Nó không phải do model tự nghĩ ra và không phải guideline của bài lab.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?: `class_id` là mã số để máy đọc, `class_name` là tên người dùng đọc được, còn `taxonomy_name` giải thích danh sách lớp này thuộc hệ phân loại nào. Nếu không lưu cả ba, ta có thể nhầm giữa `class_id` của hai taxonomy khác nhau hoặc đọc sai lớp khi đổi checkpoint.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?:
Cần nói rõ ảnh đó thuộc class nào khi có nhiều vật thể, ví dụ ưu tiên vật thể lớn nhất, trung tâm, hoặc điều kiện multi-label. Không thể dựa vào model score để “chọn nhãn” mà không có quy tắc rõ ràng.
- Vì sao model score không phải ground truth?: 
 Vì `score` chỉ là xác suất do mô hình tính từ dữ liệu huấn luyện, chưa được xác minh guideline. Ground truth là nhãn do reviewer xác nhận theo rule, còn model score chỉ dùng để xếp hạng và lọc prediction.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): 
Một ví dụ trong JSON là một dòng dữ liệu chứa `class_name`, `score`, và `bbox_xyxy = [x_min, y_min, x_max, y_max]`. `bbox_width` và `bbox_height` được suy ra từ khoảng cách giữa x và y tương ứng.
- Diễn giải vị trí box bằng lời: Box mô tả một vật thể được model phát hiện trong ảnh, với tọa độ pixel tính từ góc trên bên trái. Ví dụ, `x_min` và `y_min` là vị trí góc trên bên trái của hộp, trong khi `x_max` và `y_max` là góc dưới bên phải. Nếu `bbox_width` lớn, vật thể chiếm diện tích lớn hơn trong ảnh
- So sánh số prediction ở hai threshold: Khi giảm threshold từ 0.60 xuống 0.20, số prediction tăng lên vì model giữ nhiều box có độ tin cậy thấp hơn. Khi tăng threshold, số prediction giảm vì các box kém tin cậy bị loại. Đây khác với ground truth, vì threshold là cài đặt lọc prediction chứ không phải định nghĩa đúng/sai của nhãn
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Khi threshold thấp, reviewer sẽ thấy nhiều box hơn, trong đó có cả box bị rơi vào vùng không chắc chắn hoặc không cần label. Khi threshold cao, reviewer xét ít hơn nhưng có nguy cơ bỏ sót vật thể thật. Cả hai cách đều cần QC để xác định xem có object nào bị thiếu hay box nào bị quá rộng/quá hẹp
- Đề xuất một quy tắc box chặt:
 Box nên phủ kín phần vật thể có thể nhìn thấy, không nên quá rộng để bao thừa nền, và không nên quá hẹp đến mức cắt mất phần chính của vật thể. Quy tắc đơn giản là “bọc sát phần thấy được của object, giữ thừa tối thiểu nhưng vẫn bao cả thể tích chính”
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
cần guideline xác định có nên gán box cho phần còn thấy hay chỉ chọn phần đủ rõ để label. Nếu không có tiêu chuẩn rõ, cần escalation lên reviewer để quyết định, tránh tự ý phóng to box hoặc bỏ object
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
Ví dụ 
- Polygon bổ sung chi tiết gì so với box?
 
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn bộ ảnh, ví dụ một `class_name` trong taxonomy `ImageNet-1K` | Một ảnh có nhiều vật thể có thể khiến nhãn cấp ảnh mơ hồ; model score không chứng minh đúng nhãn | Dùng guideline để chọn lớp phù hợp cho ảnh, không dựa vào `score` như ground truth | Kiểm tra class list, mẫu, và có đồng ý với rule chọn lớp khi ảnh phức tạp |
| Phát hiện vật thể | Một `class_name` và một `bbox_xyxy` cho mỗi object instance | Box có thể quá rộng, quá hẹp, hoặc có vật thể bị bỏ sót; threshold thấp/high ảnh hưởng số prediction | Gắn đúng class và box theo quy tắc box chặt, không nhầm object với background | Kiểm tra box chặt, đúng object và đủ coverage; reject nếu object bị cắt/bị che không có guideline |
| Instance segmentation | Một `instance_id`, `class_name`, `bbox_xyxy`, và `polygon_xy` cho mỗi instance | Biên mask có thể mờ, tiếp xúc, hoặc che khuất; polygon dễ bị quá rộng/nhầm với box | Gán instance riêng, vẽ mask sát biên, giữ rõ ràng từng đối tượng | QC biên mask, instance separation, và các trường hợp mơ hồ cần escalation |


## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ dùng ảnh công khai có sẵn trong notebook, không tải dữ liệu cá nhân, dữ liệu nội bộ, khuôn mặt, biển số, hoặc hình ảnh nhạy cảm lên Colab/GitHub công khai
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach/GV hoặc người quản lý bài lab ngay lập tức đồng thời không tiếp tục upload dữ liệu đó lên repository công khai.
## 6. Danh sách bằng chứng

- [x] `classification_predictions.json` 
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
