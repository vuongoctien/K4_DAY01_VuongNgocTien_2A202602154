# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): <br/>
( 
    `class_id`: 468,
    `class_name`: "cab",
    `rank`: 1,
    `score`: 0.510915 (~51.09%),
    `taxonomy_name`: "ImageNet-1K"
)
- Record này mô tả toàn ảnh như thế nào?<br/>
    _Record mô tả tổng quan bức ảnh, đánh giá chủ thể chính là xe taxi ("cab") với độ tin cậy 0.510915 (~51.09%), cũng là class có độ tin cậy cao nhất (rank 1)_
- Ai định nghĩa class list mà checkpoint có thể dự đoán?<br/>
    _Tập dữ liệu ImageNet-1K định nghĩa check list_
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?<br/>
    _ID giúp máy truy xuất, xử lý và tính toán nhanh
    Tên lớp tuy không bắt buộc với máy tính, nhưng giúp con người dễ hiểu
    Taxonomy là bộ dữ liệu quy chuẩn, trường ID lấy từ đó mà ra nên đương nhiên không thể thiếu_
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?<br/>
    _Guideline cần làm rõ tiêu chuẩn chọn lớp như diện tích, ở gần trung tâm, ngữ cảnh dự án, độ rõ nét, xử lý ngoại lệ, v.v._
- Vì sao model score không phải ground truth?<br/>
    _Model score là câu chuyện của xác suất, còn ground truth là sự đúng/sai tuyệt đối.
    Kể cả model score có bằng 0 hoặc 1 thì cũng là do cách tính của thuật toán chứ không phải do kiểm chứng, nên không phải ground truth._

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):<br/>
_{
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
},_
- Diễn giải vị trí box bằng lời:<br/>
    _Góc trên - trái của box có tọa độ (x,y)= (385.33, 69.24) (cách mép trái ảnh 385.33 px và cách mép trên ảnh 69.24 px)
    Tương tự, góc dưới - phải của box có tọa độ (498.92, 348.92)
    Kích thước của box là chiều ngang * chiều dọc = 113.58 * 279.68 px^2_
- So sánh số prediction ở hai threshold: <br/>
    _Ở ngưỡng threshold thấp hơn, mô hình phát hiện ra nhiều đối tượng hơn (bao gồm cả các vật thể nhỏ hoặc mờ). Khi tăng ngưỡng threshold, số lượng prediction giảm đi do các dự đoán có độ tin cậy thấp bị lọại bỏ._
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?<br/>
    _Khi tăng threshold, độ bao phủ giảm xuống do các dự đoán có độ tin cậy thấp bị loại bỏ, giảm tải khối lượng công việc cho reviewer_
- Đề xuất một quy tắc box chặt:<br/>
    _Ví dụ với người: Box phải tính từ đỉnh đầu xuống đến tận gót giày, từ mép tay bên trái sang mép tay bên phải. Không tính phần đổ bóng._
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?<br/>
    _Với đối tượng bị che khuất hoặc cắt mép, Guideline cần quy định rõ:
    1, Tỷ lệ hiển thị tối thiểu (ví dụ: chỉ gán nhãn khi nhìn thấy 20% vật thể)
    2, Quy tắc vẽ box(chỉ bao phủ phần pixel nhìn thấy được hay ước lượng cả phần bị che)
    3, Có thể tạo thêm thuộc tính occluded (kiểu bit) để biểu thị cho việc bị che khuất<br/>
Escalation: Nếu gặp tình huống đặc biệt khó, người gán nhãn cần báo cho cấp trên._

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):<br/>
_{
    "instance_id": "traffic-001",
    "class_name": "bus",
    "score": 0.925745,
    số điểm: 120, 
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      v.v.
    ]
  },_
- Polygon bổ sung chi tiết gì so với box?<br/>
    _Polygon bổ sung thông tin chính xác về hình dạng thực sự của đối tượng, loại bỏ hoàn toàn các điểm thuộc background hoặc vật thể khác bị lẫn vào bên trong bounding Box._
- `instance_id` dùng để làm gì và không phải loại ID nào?<br/>
    _instance_id: Dùng để định danh từng đối tượng, kể cả chung một lớp, ví dụ có 1 con gà và 2 ô tô, instance_id không phải là class_id (định danh lớp)_
- Đề xuất một quy tắc biên mask:<br/>
    _Ví dụ: Đường biên mask phải ôm sát ranh giới pixel ngoài cùng của vùng nhìn thấy được với sai số không quá 10 pixel_
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?<br/>
    _Guideline cần quy định rõ quy tắc:<br/>
    1. Với vùng mờ: Lấy viền ở trung tâm vệt mờ hoặc chọn vùng có độ nét trên 50%
    2. Với vùng tiếp xúc: sử dụng đường nối tiếp xúc tự nhiên nhất
    3. Che khuất: Nếu đối tượng A bị đối tượng B cắt làm 2 phần thì gộp thành 1 hay tách thành 2 instance độc lập. Dặt diện tích hiện thị tối thiểu.<br/>
    Escalation: Nếu gặp tình huống đặc biệt khó, người gán nhãn cần báo cho cấp trên._


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Lớp đơn (Class Label / One-hot Vector) | Ảnh chứa nhiều chủ thể có độ nổi bật tương đương nhau | Chọn nhãn chủ thể chính theo quy tắc của guideline | Kiểm tra xem nhãn được gán có đúng quy tắc ưu tiên chủ thể không |
| Phát hiện vật thể | Bounding Box tọa độ `[x_min, y_min, x_max, y_max]` + `class_id` | Lẫn lộn giữa các lớp xe (ví dụ: `bus`, `cab`, `car`) khi bị che khuất | Vẽ box ôm sát vật thể và gán đúng nhãn lớp | Kiểm tra độ ôm sát của box, phát hiện box bỏ sót hoặc vẽ dư |
| Instance segmentation | Tập hợp tọa độ Polygon `[[x1, y1], [x2, y2], ...]` + `instance_id` | Khó xác định ranh giới ở các vùng bị bóng râm hoặc mờ nhòe | Chấm các điểm polygon men theo đúng viền thực của đối tượng | So sánh độ khớp của mask so với biên thực tế của thực thể |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: _Không thu thập, lưu trữ hoặc chia sẻ thông tin nhận dạng cá nhân trong các tập dữ liệu huấn luyện._
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: _Quản lý dự án (Project Manager / Team Lead) hoặc Giảng viên / Ban tổ chức chương trình."_

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
