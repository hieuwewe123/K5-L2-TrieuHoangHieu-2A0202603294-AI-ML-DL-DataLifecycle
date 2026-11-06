# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:06-11-2026**

**Runtime Colab:** GPU-T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh (Image Classification)

- **Tên tệp ảnh đã phân tích:** `traffic.jpg` (COCO Image ID: 210273)
- **Kết quả dự đoán có độ tin cậy cao nhất (Rank 1):**
  - Lớp dự đoán (`class_name`): `cab`
  - Độ tin cậy (`score`): `0.510915`
- **Lớp dự đoán tiếp theo (Rank 2):**
  - Lớp dự đoán (`class_name`): `minibus`
  - Độ tin cậy (`score`): `0.164284`
- **Đánh giá & suy luận về Taxonomy:**
  - Mặc dù đây là một cảnh đường phố thực tế ở Vũ Hán (Wuhan) với rất nhiều xe buýt, xe con và phương tiện hỗn hợp giao thoa, mô hình ImageNet-1K gán nhãn hàng đầu là `cab` (taxi) và `minibus` do taxonomy của ImageNet-1K tập trung cao vào các đặc tính phương tiện chở khách thông dụng trên phố.
  - Điều này giải thích tại sao mô hình không gán nhãn chung chung là "traffic" hay "street" mà cố gắng tìm một lớp cụ thể nhất đại diện cho diện mạo xe cộ chiếm ưu thế.

---

## 2. Phát hiện vật thể (Object Detection)

- **Tên tệp ảnh đã phân tích:** `kitchen.jpg` (COCO Image ID: 397133)
- **So sánh số lượng bounding box phát hiện được ở các ngưỡng khác nhau:**
  - Với ngưỡng `threshold=0.20`: Phát hiện **17 vật thể** bao gồm các lớp: `person`, `bowl`, `oven`, `cup`, `spoon`, `potted plant`, `dining table`, `bottle`.
  - Với ngưỡng `threshold=0.35` (mặc định): Phát hiện **11 vật thể** chủ yếu gồm: `person`, `bowl`, `oven`, `cup`.
  - Với ngưỡng `threshold=0.60`: Phát hiện chỉ còn **6 vật thể** chính: `person`, `bowl`, `oven`.
- **Hiện tượng rút ra khi tăng ngưỡng threshold:**
  - Khi tăng ngưỡng threshold từ thấp lên cao, các dự đoán có độ tự tin (confidence score) thấp (như chiếc thìa nhỏ `spoon` hay cốc `cup`) sẽ bị lọc bỏ hoàn toàn. Điều này làm giảm tỷ lệ báo động giả (False Positives) nhưng đồng thời lại tăng tỷ lệ bỏ sót vật thể thực tế trong cảnh (False Negatives).
- **Nhận diện lỗi gán nhãn thực tế của mô hình (Evidence & Rationale):**
  - Trên ảnh `detection_predictions.png`, mô hình YOLO11n gặp lỗi nhận diện và gán nhãn một số chiếc đĩa phẳng hoặc khuôn nướng trên bàn thành lớp `bowl` (bát), hoặc bỏ sót một vài chiếc thìa inox treo nhỏ ở thanh treo phía trên do kích thước nhỏ và bị lẫn vào hậu cảnh.
- **Quy tắc gán nhãn đề xuất (Guideline) cho Annotator:**
  - **Quy tắc gộp/tách đĩa và bát:** Chỉ gán nhãn `bowl` cho các vật dụng chứa dạng lòng sâu, có chiều cao thành tối thiểu bằng 1/3 đường kính miệng. Các đĩa phẳng hoặc khuôn nướng dẹt phải được gán nhãn là `plate` hoặc bỏ qua nếu không thuộc danh mục phân loại.
  - **Quy tắc vật thể bị che khuất một phần (Occlusion):** Nếu một vật thể bị che khuất dưới 50% diện tích bề mặt, annotator vẫn phải vẽ bounding box bao quanh toàn bộ kích thước ước lượng của vật thể đó. Nếu che khuất trên 50%, tiến hành báo cáo lên Reviewer để cân nhắc loại bỏ hoặc gán nhãn đặc biệt.
- **Quy trình Escalation khi gặp ca khó gán nhãn:**
  - Khi annotator không thể phân biệt rõ vật thể đó là bát hay đĩa (do góc chụp tối hoặc bị biến dạng), tuyệt đối không tự ý đoán mò. Phải gắn cờ nghi vấn (flag) kèm ghi chú mô tả và chuyển tiếp trực tiếp lên Reviewer/Lab Coach thông qua kênh Slack dự án để thống nhất ý kiến gán nhãn.

---

## 3. Phân đoạn đối tượng (Instance Segmentation)

- **Tên tệp ảnh đã phân tích:** `kitchen.jpg` (COCO Image ID: 397133)
- **Sự khác biệt thực tế giữa Bounding Box và Đa giác (Polygon):**
  - Bounding box là một hình chữ nhật bao quanh vật thể, vô tình chứa rất nhiều pixel nền không mong muốn (ví dụ: khoảng trống giữa hai chân của người đầu bếp, hoặc khoảng trống xung quanh thân chiếc thìa treo).
  - Đa giác (Polygon) bám sát theo từng đường cong biên thực tế của cơ thể người, dụng cụ nấu ăn, giúp định vị chính xác tuyệt đối vùng pixel thực của đối tượng.
- **Tình huống bắt buộc dùng Phân đoạn đối tượng (Instance Segmentation):**
  - Đối với các vật thể có biên dạng phức tạp hoặc xếp chồng chéo lên nhau (ví dụ: chiếc thìa nấu ăn mỏng, tay cầm của chảo, hoặc quần áo người đầu bếp đang đứng nghiêng), việc sử dụng đa giác là bắt buộc để tránh chồng lấn nhãn giữa các vật thể kề nhau và bảo toàn hình học thực tế cho mô hình học máy phân đoạn.

---

## 4. Thiết kế quy trình QC (Quality Control)

Bảng phân công vai trò rõ ràng để tránh xung đột lợi ích:

| Vai trò | Nhiệm vụ cụ thể | Tiêu chí chấp nhận sản phẩm (Acceptance Criteria) |
| :--- | :--- | :--- |
| **Annotator** (Người gán nhãn) | - Đọc kỹ guideline gán nhãn.<br>- Thực hiện vẽ box/polygon trực tiếp trên công cụ CVAT.<br>- Ghi chú các ca khó phân loại. | - 100% các vật thể rõ ràng được gắn nhãn đúng lớp.<br>- Sai lệch tọa độ hộp/polygon không vượt quá 3 pixel so với biên thực tế. |
| **Reviewer** (Người kiểm duyệt) | - Kiểm tra chéo tối thiểu 20% lượng ảnh ngẫu nhiên từ annotator.<br>- Sửa đổi trực tiếp các lỗi nhỏ hoặc reject toàn bộ batch nếu tỷ lệ lỗi vượt quá mức cho phép.<br>- Giải quyết các ca khó được annotator gắn cờ. | - Tỷ lệ lỗi gán nhãn (nhầm lớp, vẽ lệch nghiêm trọng) của batch dưới 2% để phê duyệt xuất dữ liệu.<br>- Đảm bảo tất cả các escalation đều được phản hồi thống nhất. |

---

## 5. Quy tắc an toàn dữ liệu và Bảo mật (Data Security)

- **Quy tắc 1: Tuyệt đối không tải dữ liệu lên công cụ công cộng:** Không bao giờ upload hình ảnh y tế, dữ liệu khách hàng hoặc hình ảnh camera an ninh nội bộ lên các trang web chuyển đổi định dạng trực tuyến tự do, các AI chatbot công cộng hoặc các server không được doanh nghiệp cấp phép an toàn.
- **Quy tắc 2: Chặn rò rỉ API Key và Token:** Không được lưu trực tiếp các API

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
