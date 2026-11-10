# Kiểm tra dữ liệu ViHSD — 10/11/2026

Người dùng cung cấp ba CSV tải từ [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd), hiện đặt tại `data/train.csv`, `data/dev.csv`, `data/test.csv`. Chỉ kiểm tra schema và thống kê local, **không fit/train**. Không sửa CSV gốc, không chép bình luận vào báo cáo/source/ZIP. Chưa biết SHA commit Hugging Face của lần tải, nên dùng SHA-256 từng file để cố định snapshot thay vì tự điền revision.

## Schema và mapping

- Cột văn bản: `free_text`.
- Cột nhãn: `label_id`, có đúng ba mã `0`, `1`, `2`, không có nhãn rỗng/lạ.
- Mapping: `0 → CLEAN`, `1 → OFFENSIVE`, `2 → HATE`.
- Căn cứ mapping: [notebook DNN trong repo tác giả](https://github.com/sonlam1102/vihsd/blob/main/ViHSD_DNN.ipynb) đọc `free_text`/`label_id`, đưa nhãn trực tiếp vào `to_categorical(num_classes=3)`, tính confusion matrix và đặt thứ tự hàng/cột clean/offensive/hate. Đây là mapping đối chiếu từ luồng code tác giả, không suy ra bằng cách đoán nội dung bình luận.

Preset máy đọc được: `configs/dataset.example.json`. `scripts/build_notebooks.py` đưa cùng preset này vào cell cấu hình của cả ba notebook.

## Số mẫu

| Split | Gốc | Văn bản rỗng sau chuẩn hóa | Dùng full | CLEAN sau lọc | OFFENSIVE sau lọc | HATE sau lọc |
|---|---:|---:|---:|---:|---:|---:|
| train | 24.048 | 2 | 24.046 | 19.885 | 1.605 | 2.556 |
| dev | 2.672 | 0 | 2.672 | 2.190 | 212 | 270 |
| test | 6.680 | 0 | 6.680 | 5.548 | 444 | 688 |

Số lớp train gốc trước lọc: CLEAN 19.886, OFFENSIVE 1.606, HATE 2.556. Train có hai bản ghi văn bản rỗng, tại số bản ghi CSV **10.952** và **20.882** (header tính là bản ghi 1). Đây là chỉ số bản ghi CSV, không nhất thiết là dòng vật lý vì một trường CSV có thể chứa xuống dòng.

## Quy tắc chung cho ba notebook

1. Kiểm tra SHA-256 đúng snapshot, cột và mapping. Nhãn lạ vẫn gây lỗi ngay cả khi văn bản rỗng.
2. Chuẩn hóa Unicode NFC và khoảng trắng.
3. Preset chọn `empty_text_policy="drop"`: bỏ văn bản rỗng khi đọc vào bộ nhớ, ghi số lượng và số bản ghi vào `dropped_empty_text`. API `load_data` mặc định vẫn là `error` nếu caller không chọn quy tắc rõ ràng.
4. Không sửa file nguồn, không chia lại hoặc chuyển mẫu giữa train/dev/test.
5. Giữ các dòng trùng, chỉ báo thống kê. Không tự bỏ trùng theo cách riêng của từng thành viên.
6. Metadata ghi riêng `original_counts`, `filtered_counts`, `used_counts`; smoke dùng số mẫu con sau bước lọc, không phải toàn bộ số full ở bảng trên.

Class weights, vocabulary và TF-IDF chỉ tính từ train sau lọc. Dev chọn cấu hình/checkpoint; test chỉ đánh giá sau khi đã cố định lựa chọn. Test không có mẫu rỗng nên giữ đủ 6.680 mẫu.

## Trùng văn bản giữa các split

So sánh văn bản sau NFC + chuẩn hóa khoảng trắng, có phân biệt hoa/thường. Đếm sau khi bỏ văn bản rỗng, trước tiền xử lý riêng từng model. Mỗi cặp được đếm độc lập; một văn bản có thể xuất hiện ở cả ba split nên không cộng các cặp thành số duy nhất toàn bộ dataset.

| Cặp split | Văn bản duy nhất trùng | Số dòng liên quan ở split thứ nhất | Số dòng liên quan ở split thứ hai |
|---|---:|---:|---:|
| train – dev | 321 | 351 | 324 |
| train – test | 787 | 840 | 804 |
| dev – test | 91 | 95 | 94 |

Các dòng trùng có thể làm điểm dev/test cao hơn khả năng tổng quát hóa trên bình luận mới. Báo cáo kết quả cần nêu hạn chế này và chính sách giữ nguyên split. Chưa có kết quả model nên chưa định lượng ảnh hưởng lên điểm số.

## SHA-256 CSV gốc

```text
train.csv 516a4ac12bcfd64720a6de352b3c30d019377f90a5907eff80e91d0ae2526360
dev.csv   1d42bc7e78b796f9a61bb30490171e5d48cbdee41f97c1235754894d28402ff7
test.csv  ba8ae9b62c40caebcbe8af9a09590296bca286e83e22dd639de31a62ed16f5a5
```

Tái kiểm tra từ root dự án:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_data.py --data-dir data
```

Lệnh chỉ in metadata thống kê, không in bình luận và không huấn luyện. Nếu SHA khác, dừng để cả nhóm kiểm tra snapshot; không ghi đè preset bằng hash mới mà chưa kiểm tra lại.

## Chạy Colab với bộ dữ liệu này

Chạy `python scripts/package_source.py` để tạo ZIP source mới, rồi mở một trong ba notebook mới ở `notebooks/`. Giữ `SOURCE_MODE="zip"`, `DATA_MODE="csv"`, `UPLOAD_CSV=True` và các trường dataset đã điền sẵn. Run all, upload `vihsd-source.zip`, sau đó chọn cùng lúc ba CSV gốc. Không gói CSV vào ZIP source. `dev.csv` đúng tên đang được code yêu cầu, không cần đổi thành validate/validation.

Không dùng notebook mới với source ở commit cũ thiếu hỗ trợ `empty_text_policy`/`expected_sha256`. Khi dùng Git thay ZIP, source cần là revision mới đã chứa các thay đổi này, dùng cùng SHA cho cả nhóm. Notebook 03 vẫn cần điền SHA checkpoint PhoBERT riêng. Smoke trước, full sau với ID run mới; chưa có kiểm chứng Run all Colab hay artifact thật trong lần kiểm tra dữ liệu này.
