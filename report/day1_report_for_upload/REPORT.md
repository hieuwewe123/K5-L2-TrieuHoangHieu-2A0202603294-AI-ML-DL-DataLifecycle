# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06-11-2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không

> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output.

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `day1_lab_outputs/visuals/classification_top5.png`.
**Sample:** `traffic` (COCO Image ID `210273`).

1. Record hạng 1: `class_id=468`; `class_name=cab`; `rank=1`; `score=0.510915`; `taxonomy_name=ImageNet-1K`.
2. Record này nói về **cả bức ảnh**, không phải một vật thể riêng. Đây là tác vụ image classification: các lớp được xếp hạng cho ảnh `traffic`; record không có bounding box hay polygon để xác định vị trí của từng vật thể.
3. Danh sách lớp đến từ taxonomy gắn với checkpoint (`ImageNet-1K`), không phải model tự tạo tên lớp.
4. `class_id` là mã lớp trong taxonomy, `class_name` dễ đọc cho người, còn `taxonomy_name` cho biết mã và tên đó thuộc danh sách lớp nào. Chỉ ghi tên lớp có thể gây nhầm giữa các taxonomy.
5. Guideline cần xác định đơn vị phân loại và quy tắc chọn lớp khi ảnh có nhiều chủ thể (ví dụ: lớp của chủ thể chính hay nhãn đa lớp). Không nên tự chọn nếu guideline chưa quy định.
6. `score=0.510915` là mức độ tin cậy/xếp hạng của model, không phải xác nhận của con người. Ground truth phải được tạo theo guideline và kiểm tra, không suy ra chỉ từ score.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `day1_lab_outputs/visuals/detection_predictions.png`.
**Sample:** `kitchen` (COCO Image ID `397133`).

1. Một record: `class_name=person`; `score=0.912625`; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`; `bbox_width=113.58`; `bbox_height=279.68`.
2. Tọa độ dùng pixel, gốc ở góc trên-trái ảnh. Box bắt đầu tại `(385.33, 69.24)` và kết thúc tại `(498.92, 348.92)`, rộng khoảng `113.58` pixel và cao khoảng `279.68` pixel.
3. Theo kết quả so sánh trong notebook, số prediction lần lượt là `17` ở threshold `0.20`, `11` ở `0.35` và `6` ở `0.60`.
4. Khi hạ threshold, reviewer cần kiểm tra thêm các prediction điểm thấp: xác nhận vật thể/lớp, tìm dự đoán trùng hoặc sai, và đối chiếu vật thể có thể bị bỏ sót. Threshold chỉ lọc prediction, không phải quy tắc tạo ground truth.
5. Quy tắc box đề xuất: vẽ một hình chữ nhật sát vùng vật thể theo quy ước của dự án, không gộp vật thể bên cạnh vào cùng box; trường hợp che khuất phải xử lý theo guideline đã thống nhất.
6. Ảnh có một phần người/vật thể bị cắt ở mép trái. Nếu guideline chưa quy định cách gán nhãn vật thể bị cắt mép, annotator nên ghi chú và hỏi reviewer/Lab Coach thay vì tự đặt quy tắc.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `day1_lab_outputs/visuals/segmentation_prediction.png`.
**Sample:** `kitchen`.

1. Record được chọn: `instance_id=kitchen-001`; `class_name=person`; `score=0.899318`; `polygon_point_count=348`; năm điểm đầu của `polygon_xy`: `[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]`.
2. Box cho biết vùng chữ nhật bao quanh; polygon mô tả đường biên mask chi tiết hơn, giúp phân biệt vùng vật thể với nền bên trong box.
3. `instance_id` định danh một instance trong output của sample này. Nó không biểu thị loại lớp (class ID), cũng không theo dõi cùng vật thể qua nhiều khung hình (tracking ID).
4. Quy tắc mask đề xuất: vẽ theo biên phần vật thể nhìn thấy, tránh lấy pixel của nền hoặc vật thể liền kề; nếu biên bị che và guideline chưa nói rõ thì cần escalation.
5. Trong hình, mask của bàn nằm gần nhiều đồ vật đặt trên mặt bàn nên có vùng biên cần kiểm tra để bảo đảm mask theo đúng đối tượng. Nếu không thể phân biệt rõ ranh giới, annotator nên ghi chú và hỏi reviewer/Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ quan sát prediction. Vòng đời production được minh họa như sau:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth (đơn vị nhãn) | Lỗi hoặc điểm khó quan sát | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một hoặc nhiều nhãn cho ảnh, tùy guideline | Top-1 `cab` (score `0.510915`) là prediction cho toàn ảnh, không xác nhận mọi xe trong ảnh là taxi | Áp dụng quy tắc chọn lớp/chủ thể chính; hỏi khi guideline chưa rõ | Kiểm tra nhãn có theo taxonomy và guideline không |
| Phát hiện vật thể | Một lớp và một box cho mỗi vật thể | Có box `bowl` quanh các vật thể trên bàn cần đối chiếu bằng mắt; threshold khác nhau cho số prediction khác nhau | Gán từng vật thể theo quy tắc box; ghi chú vật thể bị che/cắt hoặc lớp mơ hồ | Kiểm tra lớp, vị trí box, vật thể trùng/sai và ca bị bỏ sót |
| Instance segmentation | Một lớp và mask/polygon riêng cho mỗi instance | Ranh giới của bàn và các vật đặt trên bàn có thể khó tách | Vẽ mask theo guideline và giữ riêng từng instance | Kiểm tra biên mask, vùng nền/vật lân cận bị lấy nhầm và tính nhất quán giữa instance |

Model score là bằng chứng về prediction, không phải tiêu chí chất lượng ground truth. Ca guideline chưa bao phủ cần được chuyển reviewer/Lab Coach để thống nhất; không tự bịa nhãn.

## 5. An toàn dữ liệu

1. Chỉ dùng ảnh công khai được notebook cung cấp; không tải dữ liệu cá nhân, khách hàng hoặc dữ liệu nội bộ lên công cụ công cộng, và không đưa API key/token vào repo, notebook hay log.
2. Nếu gặp dữ liệu ngoài phạm vi, dừng xử lý và báo Lab Coach/giảng viên để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
