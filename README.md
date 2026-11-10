# NLP501_Vietnamese_Comment_Classification

Phân loại bình luận tiếng Việt với ViHSD.

Demo học tập cho nhóm ba người, dự đoán **CLEAN** (bình thường), **OFFENSIVE** (xúc phạm), **HATE** (thù ghét). Mỗi người train một model trên Google Colab. Local chỉ kiểm tra mã và chạy suy luận bằng một artifact đã xuất.

**Trạng thái bàn giao:** đã triển khai mã, ba notebook và app. Chưa chạy Colab, chưa có CSV được cấp quyền hoặc artifact do nhóm cung cấp. Chưa có điểm thực nghiệm, checkpoint hay model được chọn. Các số liệu giả lập trong tests chỉ dùng kiểm tra hợp đồng và không phải kết quả nghiên cứu.

| Model | Thành viên | Dev macro-F1 | Test macro-F1 | Trạng thái |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 1 | — | — | Chờ kết quả Colab |
| TextCNN | 2 | — | — | Chờ kết quả Colab |
| PhoBERT | 3 | — | — | Chờ kết quả Colab |

## Cài đặt local

Đã kiểm tra trên Windows, **Python 3.13.12**. Package khai báo hỗ trợ Python 3.10–3.13; các phiên bản Python khác chưa chạy kiểm tra ở lần bàn giao này. Không có lệnh train local trong hướng dẫn này.

Chạy từ root dự án bằng PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python -m pip install -e .
python -m streamlit run app.py
```

Nếu PowerShell không cho activate, dùng trực tiếp `.\.venv\Scripts\python.exe` thay cho `python`. Trên Linux/macOS: `python3 -m venv .venv` rồi `source .venv/bin/activate`, sau đó chạy các lệnh pip tương tự.

Mặc định local chỉ cài backend scikit-learn. Khi dùng **TextCNN hoặc PhoBERT**, cài thêm:

```powershell
python -m pip install -r requirements-neural.txt
```

Colab cài `requirements-colab.txt`, kế thừa cùng phiên bản backend với local: NumPy 2.2.6, scikit-learn 1.6.1, joblib 1.4.2, PyTorch 2.7.1, Transformers 4.51.3, safetensors 0.5.3, PyVi 0.1.1. Manifest ghi phiên bản dùng lúc train; loader yêu cầu khớp phiên bản đã ghi. Dependency chỉ phục vụ notebook không bắt buộc cho app baseline.

## Cấu trúc

```text
app.py                              # Một giao diện Streamlit tiếng Việt
pyproject.toml                      # Package vihsd, layout src
requirements-{local,neural,colab,dev}.txt
configs/active_model.example.json   # Cấu hình mẫu được giữ trong Git
notebooks/
  01_tfidf_logistic_regression.ipynb
  02_textcnn.ipynb
  03_phobert.ipynb
src/vihsd/
  labels.py                         # 0=CLEAN, 1=OFFENSIVE, 2=HATE, chỉ là mã NỘI BỘ
  data.py                           # CSV, mapping, SHA-256, kiểm tra split
  evaluation.py                     # Báo cáo theo thứ tự nhãn cố định
  training.py                       # Huấn luyện bị giới hạn ở Colab
  artifacts.py                      # Manifest, phiên bản và đường dẫn file
  inference.py                      # Nạp đúng một backend trên CPU
  comparison.py                     # Kiểm tra và chọn bằng dev
  models/{classical,textcnn,phobert}.py
scripts/compare_models.py
scripts/build_notebooks.py          # Tái tạo đúng ba notebook
tests/test_project.py
artifacts/.gitkeep                  # Chưa có model thật
```

CSV, ZIP, artifact, cấu hình active, secrets, môi trường ảo và cache nằm trong `.gitignore`. Không đưa bình luận nguồn vào mã, notebook outputs hay Git.

## Dữ liệu và mapping

Nguồn ưu tiên là [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd). Trang nguồn yêu cầu đăng nhập và chấp thuận điều kiện để đọc file. Xin quyền bằng tài khoản của thành viên; không dùng mirror hoặc dataset thay thế. Tham khảo [repo tác giả](https://github.com/sonlam1102/vihsd) và bài báo [A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts](https://arxiv.org/abs/2103.11528), Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen, 2021. Kiểm tra lại liên kết bài báo trong repo tác giả nếu nhà xuất bản đổi URL.

Mỗi notebook hỗ trợ hai cách:

1. **HF có quyền:** tạo Colab Secret `HF_TOKEN`, bật quyền notebook đọc Secret. Đặt `DATA_MODE="hf"`, điền `HF_REVISION` là SHA commit đầy đủ của dataset và `HF_FILENAMES` là đường dẫn ba CSV thực tế ở revision đó. Chỉ tải từ `uitnlp/vihsd`. Không dán token vào cell, URL clone hay log.
2. **CSV hợp lệ:** đặt `DATA_MODE="csv"`, upload đồng thời `train.csv`, `dev.csv`, `test.csv`. Hoặc đặt `UPLOAD_CSV=False`, sửa `CSV_DIR` sang thư mục đã có đủ ba file trên Colab/Drive. Điền `CSV_SOURCE` ghi nguồn cấp hợp lệ.

Code không mặc định đoán tên cột hoặc mã nhãn nguồn. Trong cell cấu hình, điền `TEXT_COLUMN`, `LABEL_COLUMN`, `LABEL_MAPPING` và `MAPPING_SOURCE` sau khi đọc schema/tài liệu nguồn được cấp quyền. Các giá trị mapping phải là đúng ba nhãn CLEAN/OFFENSIVE/HATE. Ví dụ minh họa **chỉ khi CSV thực sự dùng tên nhãn**:

```python
TEXT_COLUMN = "ten_cot_da_xac_minh"
LABEL_COLUMN = "ten_cot_nhan_da_xac_minh"
LABEL_MAPPING = {"CLEAN": "CLEAN", "OFFENSIVE": "OFFENSIVE", "HATE": "HATE"}
MAPPING_SOURCE = "Tài liệu nguồn cấp: vị trí xác nhận ý nghĩa và mã nhãn"
```

Nếu nguồn dùng mã số, phải điền mapping mã số dạng chuỗi theo tài liệu đã xác minh, không suy ra từ thứ tự nội bộ. Tại thời điểm bàn giao chưa xác nhận schema CSV trong vùng dữ liệu có hạn chế truy cập, nên cấu hình mapping để trống và notebook dừng nếu thiếu.

Loader chuẩn hóa Unicode NFC và khoảng trắng; dừng khi dữ liệu rỗng, cột thiếu, văn bản rỗng, nhãn lạ hoặc split thiếu lớp. Giữ nguyên split gốc và mọi dòng trùng, báo số văn bản trùng và số dòng liên quan giữa từng cặp split. Chính sách này bảo toàn benchmark gốc nhưng kết quả có thể chịu ảnh hưởng bởi trùng dữ liệu; ghi rõ trong báo cáo. Ba người phải dùng cùng chính sách, không tự bỏ trùng riêng từng notebook.

Manifest ghi nguồn, revision HF hoặc SHA-256 từng CSV, cột và mapping, số mẫu từng lớp, số dòng trùng, seed, quy tắc chuẩn hóa/lọc và số mẫu thực dùng. Script so sánh yêu cầu metadata dataset giống nhau, vì vậy cả nhóm nên dùng cùng phương thức lấy dữ liệu và cùng giá trị cấu hình nguồn/mapping.

## Chạy ba notebook trên Colab

Phân công: thành viên 1 giải thích TF-IDF/ngram và regularization C; thành viên 2 giải thích embedding, convolution, pooling, vocabulary và padding; thành viên 3 giải thích tách từ, tokenizer PhoBERT và fine-tune. Cả ba cùng phân tích confusion matrix và sai số từng lớp.

Chuẩn bị một ZIP source chung cho nhóm từ root, giữ đúng cấu trúc `src/vihsd`:

```powershell
python scripts/package_source.py
```

Script tạo `vihsd-source.zip` theo danh sách source cố định, chỉ lấy cấu hình mẫu và không lấy secrets/artifacts. Chia sẻ đúng ZIP này cho ba người; notebook ghi SHA-256 ZIP vào metadata code. Hoặc đưa source lên Git repo của nhóm, đặt `SOURCE_MODE="git"`, `REPO_URL` và `REPO_REVISION` là SHA 40 ký tự. Notebook checkout detached revision; không mặc định clone nhánh thay đổi theo thời gian.

Cho từng notebook:

1. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** rồi tải một trong ba file `notebooks/*.ipynb`.
2. Sửa cell **Cấu hình**: nguồn code, nguồn dữ liệu, cột/mapping đã xác minh, seed và ID lần train. Ba notebook dùng cùng seed 42, cùng source ZIP/SHA và cùng snapshot CSV. Dùng ID mới cho mỗi lần chạy, ví dụ `tfidf-full-001`, `textcnn-full-001`, `phobert-full-001`; không ghi đè thư mục cũ.
3. Notebook 01 dùng CPU; notebook 02/03 chọn **Runtime → Change runtime type → T4 GPU** nếu còn quota. Với notebook 03, điền `PHOBERT_REVISION` là SHA đầy đủ của [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2/tree/main).
4. Giữ `SMOKE=True` lần đầu, chọn **Runtime → Run all**, upload source ZIP và ba CSV khi được hỏi. Smoke lấy tối đa 32 mẫu mỗi lớp mỗi split, train neural 1 epoch; kết quả được đánh dấu smoke và bị loại khỏi chọn model.
5. Sau khi smoke chạy thành công, đặt `SMOKE=False`, dùng `RUN_ID` mới rồi Run all cho kết quả chính thức. Không điều chỉnh cấu hình theo điểm test. Nếu Colab yêu cầu restart sau pip, restart rồi Run all lại từ đầu.
6. Các cell cuối hiển thị dev/test riêng, nạp lại artifact bằng adapter CPU dùng chung với app, dự đoán ba câu tự soạn, rồi tải ZIP. Đặt `SAVE_TO_DRIVE=True` để lưu thêm ZIP vào `DRIVE_OUTPUT`; notebook yêu cầu mount Drive. Không lưu token vào Drive.

Mặc định full: TF-IDF tối đa 80.000 đặc trưng, thử C=0.5/1/2 trên dev; TextCNN 5 epoch, batch 64, 128 token; PhoBERT 3 epoch, batch 8, 128 token, AdamW 2e-5. Class weights, vocabulary và TF-IDF chỉ tính từ train. Neural chọn epoch có macro-F1 dev cao nhất, sau đó mới đánh giá test một lần. Không gộp train+dev để fit lại.

PhoBERT dùng **PyVi 0.1.1** tách từ trước tokenizer, cùng hàm `segment()` khi train và suy luận. PyVi đóng gói tài nguyên tách từ trong wheel, không cần Java và không tải thêm model khi chạy local. [Tài liệu VinAI](https://huggingface.co/vinai/phobert-base-v2) yêu cầu đầu vào đã tách từ và khuyến nghị RDRSegmenter; lựa chọn PyVi ở dự án này cần được ghi rõ như một khác biệt trong báo cáo. Không tự thay segmenter cho artifact đã train. Model và tokenizer PhoBERT được lưu bằng `save_pretrained`, trọng số safetensors; adapter chỉ nạp bằng `local_files_only=True`.

## Nhận artifact, so sánh và đổi model

Mỗi ZIP chứa trực tiếp `manifest.json`, `metrics.json` và file suy luận. Giải nén vào thư mục độc lập tương ứng, ví dụ:

```powershell
New-Item -ItemType Directory -Force artifacts\tfidf\tfidf-full-001
Expand-Archive -Path tfidf-full-001.zip -DestinationPath artifacts\tfidf\tfidf-full-001
```

Làm tương tự cho TextCNN và PhoBERT. Kiểm tra `manifest.json` nằm trực tiếp trong thư mục run, không lồng thêm một tầng tên ZIP. Chỉ dùng artifact của nhóm; joblib có thể thực thi mã khi nạp. App không có chức năng upload model.

So sánh từ root:

```powershell
python scripts/compare_models.py
python scripts/compare_models.py --write-config
```

Lệnh đầu chỉ in bảng; lệnh thứ hai mới ghi `configs/active_model.json` nếu đủ kết quả full hợp lệ của cả ba loại model trên cùng metadata dataset/split/mapping/seed. Smoke được hiển thị riêng và không được chọn. F1 từng lớp dev/test được in riêng. Chọn macro-F1 **dev** cao nhất; hòa thì artifact ít byte hơn, rồi ID từ điển tăng dần (đường dẫn là tiêu chí cuối nếu ID trùng). Test chỉ dùng báo cáo, không ảnh hưởng chọn model. Nếu thiếu hoặc không tương thích dữ liệu, trạng thái là tạm thời và cấu hình cũ không bị ghi đè. Có thể đổi root artifact bằng `--artifacts artifacts`, đổi vị trí file cấu hình bằng `--config configs/active_model.json`.

Có thể chọn thủ công một artifact của nhóm để thử app trước khi đủ ba kết quả. Sao chép cấu hình mẫu và sửa đường dẫn:

```powershell
Copy-Item configs\active_model.example.json configs\active_model.json
```

```json
{"artifact_dir": "artifacts/textcnn/textcnn-full-001"}
```

Cấu hình chỉ chứa `artifact_dir`, là đường dẫn tương đối trong root repo. Sửa thành run khác rồi khởi động lại:

```powershell
python -m streamlit run app.py
```

App nạp đúng một model trên CPU, chỉ import backend đang cần; cache chỉ giữ một entry và khóa cache gồm đường dẫn + kích thước/thời điểm sửa các file. Không phải sửa app khi đổi model. Manifest kiểm tra schema, thứ tự nhãn, tên file, tiền xử lý và phiên bản thư viện. TextCNN lưu vocabulary, PAD/UNK, max length, kiến trúc và safetensors. Điểm đầu ra được gọi là **điểm dự đoán của model**, chưa hiệu chỉnh xác suất. Bình luận có giới hạn 5.000 ký tự; văn bản rỗng bị từ chối. Đây là demo học tập, cần người kiểm tra kết quả trong ngữ cảnh.

## Khắc phục lỗi

- **401/403 khi lấy ViHSD:** đăng nhập trang dataset, chấp thuận điều kiện/xin quyền; kiểm tra Secret HF_TOKEN có quyền đọc và notebook được phép dùng. Kiểm tra SHA/tên file ở đúng revision. Nếu chưa được cấp quyền, dừng và dùng CSV mà nhóm đã được cấp hợp lệ; không lấy bản mirror.
- **Thiếu mapping/cột:** đọc schema và tài liệu nguồn, điền tên cột, mapping và nguồn xác minh. Code cố ý dừng thay vì đoán.
- **Hết quota GPU/VRAM:** xin runtime GPU khác hoặc đợi quota phục hồi; giảm batch PhoBERT 8 xuống 4/2 trước lần train mới. Smoke kiểm tra luồng không thay thế kết quả full. PhoBERT CPU có thể chạy rất lâu. Dùng cùng cấu hình giữa các lần so sánh có ý nghĩa.
- **Thiếu cấu hình/artifact:** app hiển thị hướng dẫn; chạy Colab, giải nén đúng thư mục và kiểm tra `manifest.json` cùng các file đi kèm. Chưa có model thì app chưa phân loại được.
- **Lệch phiên bản:** dùng môi trường ảo sạch, cài đúng requirements và phiên bản ghi trong manifest. Không sửa số phiên bản trong manifest để bỏ qua kiểm tra; nếu đổi dependency khi train, cài lại cùng phiên bản ở local. Môi trường Colab thực tế phải được xác nhận khi nhóm chạy notebook.
- **PhoBERT không suy luận offline:** cài sẵn requirements-neural khi có mạng; chép đủ thư mục model/tokenizer và PyVi. Adapter không tải checkpoint/tokenizer từ mạng. Không đổi `local_files_only` để che lỗi thiếu file.
- **ID run đã tồn tại:** dùng ID mới. Artifact dở dang sau lỗi không được dùng so sánh; script báo bỏ qua khi thiếu metadata hoặc file.

## Kiểm tra và giới hạn nghiệm thu

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q src scripts app.py
python -m pip check
```

Đã đạt **10 tests**, kiểm tra cấu trúc và cú pháp ba notebook (outputs trống), import package, CSV/mapping/fingerprint bằng câu tự soạn, quy tắc chọn model trên metadata tạm (dev ưu tiên, tie, thiếu model, smoke, dataset khác nhau, điểm không hợp lệ), CLI chỉ ghi cấu hình khi có cờ, kiểm tra đường dẫn và giao diện Streamlit khi thiếu model. Kiểm tra thêm bảng/biểu đồ và cache qua stub UI; đây không phải xác nhận suy luận hoặc đổi artifact thật. Các tests không gọi fit/train, không xuất checkpoint và không để lại metadata giả trong `artifacts/`.

**Cần nhóm xác nhận sau Colab:** quyền truy cập và schema ViHSD thật; Run all ba notebook; huấn luyện, điểm số và xuất ZIP; nạp lại artifact thật; suy luận local và chuyển cấu hình giữa hai artifact thật; PhoBERT offline đầy đủ. Chưa có artifact nên chưa kiểm chứng các bước này và chưa công bố model tốt nhất.

Các đường dẫn bàn giao chính: `notebooks/`, `src/vihsd/`, `scripts/compare_models.py`, `configs/active_model.example.json`, `app.py`. Báo cáo kiểm tra chi tiết ở `VALIDATION.md`.
