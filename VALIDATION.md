# Biên bản kiểm tra — 10/11/2026

Môi trường: Windows, Python 3.13.12; `.venv` riêng trong dự án. Đã cài package bằng editable install và dependencies theo `requirements-local.txt`, `requirements-neural.txt`, `requirements-dev.txt`.

| Hạng mục | Kết quả và phạm vi |
|---|---|
| `python -m pytest -q` | **14 passed**; một cảnh báo deprecation Altair phát sinh từ Streamlit 1.45.1, không gây lỗi hiển thị trong AppTest |
| Ba notebook | Đúng ba file; `nbformat.validate` đạt; mọi code cell parse được bằng AST; không có output/execution count |
| Package và backend | Import được toàn bộ module, TextCNN, AutoTokenizer và AutoModelForSequenceClassification; không tải checkpoint |
| Tiền xử lý | PyVi tách từ câu tự soạn thành công; regex TextCNN và padding/UNK kiểm tra thủ công thành công |
| CSV | Test bằng câu tự soạn: mapping bắt buộc, dòng lỗi, SHA-256, số mẫu và báo trùng split |
| Bộ CSV người dùng vừa tải | Đã đọc `data/train.csv`, `dev.csv`, `test.csv`: free_text/label_id; cố định SHA-256 và đối chiếu mapping qua code tác giả; báo cáo `DATA_REPORT.md` |
| Lọc văn bản rỗng | 2 mẫu train rỗng; preset drop chung ba notebook, ghi metadata và giữ nguyên CSV. Test từ chối nhãn lạ dù văn bản rỗng, SHA khác và chính sách lọc không hợp lệ |
| Preset notebook | Cả ba notebook chứa cùng cấu hình dataset đã kiểm tra; kiểm tra lại bằng test, không train |
| Chọn model | Metadata giả lập trong thư mục tạm: ưu tiên dev, bỏ qua test khi chọn, tie theo size rồi ID; thiếu model/smoke/dataset khác thì chưa chọn |
| CLI | Không ghi cấu hình khi không có cờ; ghi đúng khi đủ ba metadata full cùng dataset; loại điểm NaN; thư mục artifacts thật hiện trống nên báo trạng thái tạm thời |
| Artifact/config | Từ chối mapping nhãn sai và đường dẫn vượt root; kiểm tra file bắt buộc và phiên bản khi nạp |
| App khi thiếu model | Streamlit AppTest hiển thị hướng dẫn tạo cấu hình và nhận artifact, không có exception |
| App có kết quả và cache | Dùng stub UI, không dùng model: kiểm tra ô rỗng, bảng/biểu đồ điểm và thay đường dẫn làm gọi lại loader. Đây không phải xác nhận suy luận hay đổi artifact thật |
| `python -m compileall -q src scripts app.py` | Đạt |
| `python -m pip check` | No broken requirements found |
| ZIP source | Đã tạo `vihsd-source.zip`, xác nhận giữ `src/vihsd`, đúng ba notebook và cấu hình mẫu; không chứa CSV, môi trường ảo, cache, cấu hình active hoặc model |

Không fit/train/fine-tune local. Không tạo model giả, checkpoint hay điểm thực nghiệm. Fixture metadata chỉ nằm trong thư mục tạm của pytest, không đưa vào `artifacts/` và không được dùng báo cáo chất lượng.

Chưa thể xác nhận: tải trực tiếp bằng quyền/token HF của thành viên; cài đặt/Run all trong runtime Colab thực tế; train ba model; xuất và nạp lại artifact thật; điểm dev/test; suy luận local với artifact thật; chuyển giữa hai artifact thật; vận hành PhoBERT offline đầy đủ. Dữ liệu CSV đã được người dùng cung cấp và kiểm tra local, mapping đã đối chiếu code tác giả; các bước train/inference thật vẫn cần nhóm chạy Colab rồi cung cấp ZIP.

Tài liệu thao tác và phân công: `README.md`. Điểm vào: `notebooks/`, `scripts/compare_models.py`, `app.py`. Cấu hình mẫu: `configs/active_model.example.json`. Chưa tạo `configs/active_model.json` vì chưa có artifact để trỏ đến.
