# NLP501_Vietnamese_Comment_Classification

Phân loại bình luận tiếng Việt với ViHSD.

Demo học tập cho nhóm ba người, dự đoán **CLEAN** (bình thường), **OFFENSIVE** (xúc phạm), **HATE** (thù ghét). Mỗi người train một model trên Google Colab. Local chỉ kiểm tra mã và chạy suy luận bằng một artifact đã xuất.

**Trạng thái cập nhật 11/11/2026:** nhóm đã chạy thành công cả ba notebook trên Colab ở chế độ **full**, xuất ZIP artifact và cung cấp về local. Đã xác minh metadata cùng dataset/split/mapping/seed/quy tắc lọc, nạp và thử suy luận CPU với cả ba model, kiểm tra đổi model qua giao diện trong chế độ offline. Script so sánh chọn **PhoBERT theo macro-F1 dev cao nhất**. Không huấn luyện local; số liệu dưới đây lấy từ `metrics.json` của artifact thật và khớp output notebook đã chạy.

## Kết quả huấn luyện

Các lần chạy dùng seed **42**, source revision `eeb1cd0a6c8929a3bd01eab0bde976b92e45751b` và cùng snapshot CSV đã kiểm tra. Số mẫu sau lọc: **24.046 train / 2.672 dev / 6.680 test**. TextCNN/PhoBERT chạy CUDA trên Colab. Điểm theo thang 0–1, làm tròn 4 chữ số; đây là một lần chạy cho mỗi cấu hình, không phải trung bình nhiều seed.

| Model | Thành viên | Run ID | Cấu hình thực tế | Dev macro-F1 | Test macro-F1 |
|---|---:|---|---|---:|---:|
| TF-IDF + Logistic Regression | 1 | `tfidf-full-001` | Thử C=0.5/1/2; chọn **C=2.0** | 0,6099 | 0,6052 |
| TextCNN | 2 | `textcnn-full-025-001` | 25 epoch, batch 64; chọn **epoch 22** | 0,6413 | 0,6133 |
| **PhoBERT** | 3 | `phobert-full-001` | 20 epoch, batch 8; chọn **epoch 6** | **0,6701** | **0,6461** |

Checkpoint neural được chọn theo dev, không lấy mặc định epoch cuối. Test chỉ dùng báo cáo sau khi cố định checkpoint; không dùng test để chọn model. Artifact PhoBERT ghi revision pretrained `86cd7fd4c148980922ac11a2cf5e257f2ba639e1` của `vinai/phobert-base-v2`.

F1 từng lớp:

| Model | Split | CLEAN | OFFENSIVE | HATE |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | dev | 0,9122 | 0,3837 | 0,5337 |
| TF-IDF + Logistic Regression | test | 0,9098 | 0,3712 | 0,5347 |
| TextCNN | dev | 0,9290 | 0,3978 | 0,5970 |
| TextCNN | test | 0,9253 | 0,3629 | 0,5517 |
| PhoBERT | dev | 0,9374 | 0,4920 | 0,5809 |
| PhoBERT | test | 0,9358 | 0,4333 | 0,5691 |

PhoBERT có macro-F1 dev cao nhất trong ba run hợp lệ nên được chọn cho demo local. Trên test, F1 OFFENSIVE và HATE thấp hơn CLEAN; cần xem kết quả theo lớp, không chỉ nhìn một điểm tổng hợp. Dữ liệu có trùng giữa các split như mô tả ở phần dữ liệu, nên các điểm số này không chứng minh chất lượng trên mọi bình luận mới.

Notebook có log huấn luyện, bảng precision/recall/F1 và confusion matrix:

- [01 — TF-IDF + Logistic Regression đã chạy](notebooks_run/01_tfidf_logistic_regression.ipynb)
- [02 — TextCNN đã chạy](notebooks_run/02_textcnn.ipynb)
- [03 — PhoBERT đã chạy](notebooks_run/03_phobert.ipynb)

Các bản ở `notebooks/` vẫn là template không output để thành viên chạy lại. CSV, ZIP và trọng số model không được commit; clone repo mới vẫn cần nhận artifact của nhóm, cài dependencies và cấu hình model để chạy demo.

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
notebooks_run/                      # Ba notebook đã chạy, giữ log và bảng kết quả
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
artifacts/.gitkeep                  # Git chỉ giữ placeholder; artifact thật được lưu local
```

CSV, ZIP, artifact, cấu hình active, secrets, môi trường ảo và cache nằm trong `.gitignore`. Không đưa bình luận nguồn vào mã, notebook outputs hay Git.

## Dữ liệu và mapping

Nguồn ưu tiên là [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd). Trang nguồn yêu cầu đăng nhập và chấp thuận điều kiện để đọc file. Xin quyền bằng tài khoản của thành viên; không dùng mirror hoặc dataset thay thế. Tham khảo [repo tác giả](https://github.com/sonlam1102/vihsd) và bài báo [A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts](https://arxiv.org/abs/2103.11528), Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen, 2021. Kiểm tra lại liên kết bài báo trong repo tác giả nếu nhà xuất bản đổi URL.

Mỗi notebook hỗ trợ hai cách:

1. **HF có quyền:** tạo Colab Secret `HF_TOKEN`, bật quyền notebook đọc Secret. Đặt `DATA_MODE="hf"`, điền `HF_REVISION` là SHA commit đầy đủ của dataset và `HF_FILENAMES` là đường dẫn ba CSV thực tế ở revision đó. Chỉ tải từ `uitnlp/vihsd`. Không dán token vào cell, URL clone hay log.
2. **CSV hợp lệ:** đặt `DATA_MODE="csv"`, upload đồng thời `train.csv`, `dev.csv`, `test.csv`. Hoặc đặt `UPLOAD_CSV=False`, sửa `CSV_DIR` sang thư mục đã có đủ ba file trên Colab/Drive. Điền `CSV_SOURCE` ghi nguồn cấp hợp lệ.

Ba CSV nhóm vừa tải đã được kiểm tra: tên cột `free_text` và `label_id`, nhãn nguồn là `0`, `1`, `2`. Đã đối chiếu notebook của tác giả: `label_id` đưa trực tiếp vào `to_categorical` và confusion matrix có thứ tự clean/offensive/hate. Preset được lưu tại `configs/dataset.example.json` và đã điền sẵn trong **cả ba notebook**:

```python
TEXT_COLUMN = "free_text"
LABEL_COLUMN = "label_id"
LABEL_MAPPING = {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"}
EMPTY_TEXT_POLICY = "drop"
```

Nguồn đối chiếu mapping: [ViHSD_DNN.ipynb của tác giả](https://github.com/sonlam1102/vihsd/blob/main/ViHSD_DNN.ipynb). Preset kèm `MAPPING_SOURCE`, nguồn Hugging Face và SHA-256 của ba file đã kiểm tra. Không cần đổi tên `dev.csv`: đây chính là split dùng validation. `EXPECTED_SHA256` ngăn ba thành viên dùng nhầm snapshot khác. Nếu đổi phiên bản dữ liệu, kiểm tra và cập nhật preset cho cả nhóm, không bỏ kiểm tra chỉ để chạy qua lỗi.

Loader chuẩn hóa Unicode NFC và khoảng trắng; dừng khi SHA khác, cột thiếu, nhãn lạ hoặc split rỗng/thiếu lớp sau lọc. Bộ CSV thực tế có **2 văn bản rỗng ở train**; preset dùng `empty_text_policy="drop"` để bỏ chúng khi đọc, ghi số lượng và vị trí bản ghi vào metadata, không sửa file CSV nguồn. API mặc định vẫn là `error` nếu không chọn quy tắc bỏ dòng rõ ràng. Ba notebook dùng cùng chính sách này; test không có dòng rỗng nên giữ nguyên toàn bộ 6.680 mẫu test.

Giữ nguyên phân chia train/dev/test và các dòng trùng, báo số văn bản trùng và số dòng liên quan giữa từng cặp split. Có **321 / 787 / 91** văn bản sau chuẩn hóa trùng tương ứng train–dev / train–test / dev–test. Vì vậy kết quả có thể chịu ảnh hưởng bởi trùng dữ liệu; ghi rõ trong báo cáo. Không tự bỏ trùng riêng từng notebook. Chi tiết số mẫu, SHA và chính sách ở `DATA_REPORT.md`.

| Split | Số dòng gốc | Bỏ rỗng | Số dùng full |
|---|---:|---:|---:|
| train | 24.048 | 2 | 24.046 |
| dev | 2.672 | 0 | 2.672 |
| test | 6.680 | 0 | 6.680 |

Kiểm tra lại CSV ở local (chỉ đọc dữ liệu, không fit/train, không in bình luận):

```powershell
python scripts/inspect_data.py --data-dir data
```

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
2. Sửa cell **Cấu hình**: nguồn code, nguồn dữ liệu, seed và ID lần train. Cột/mapping, chính sách bỏ rỗng và SHA đã điền sẵn cho ba CSV hiện tại, không cần nhập lại. Ba notebook dùng cùng seed 42, cùng source ZIP/SHA và cùng snapshot CSV. Dùng ID mới cho mỗi lần chạy, ví dụ `tfidf-full-001`, `textcnn-full-001`, `phobert-full-001`; không ghi đè thư mục cũ.
3. Notebook 01 dùng CPU; notebook 02/03 chọn **Runtime → Change runtime type → T4 GPU** nếu còn quota. Với notebook 03, điền `PHOBERT_REVISION` là SHA đầy đủ của [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2/tree/main).
4. Giữ `SMOKE=True` lần đầu, chọn **Runtime → Run all**, upload source ZIP và ba CSV khi được hỏi. Smoke lấy tối đa 32 mẫu mỗi lớp mỗi split, train neural 1 epoch; kết quả được đánh dấu smoke và bị loại khỏi chọn model.
5. Sau khi smoke chạy thành công, đặt `SMOKE=False`, dùng `RUN_ID` mới rồi Run all cho kết quả chính thức. Không điều chỉnh cấu hình theo điểm test. Nếu Colab yêu cầu restart sau pip, restart rồi Run all lại từ đầu.
6. Các cell cuối hiển thị dev/test riêng, nạp lại artifact bằng adapter CPU dùng chung với app, dự đoán ba câu tự soạn, rồi tải ZIP. Đặt `SAVE_TO_DRIVE=True` để lưu thêm ZIP vào `DRIVE_OUTPUT`; notebook yêu cầu mount Drive. Không lưu token vào Drive.

Mặc định full: TF-IDF tối đa 80.000 đặc trưng, thử C=0.5/1/2 trên dev; TextCNN 5 epoch, batch 64, 128 token; PhoBERT 3 epoch, batch 8, 128 token, AdamW 2e-5. Class weights, vocabulary và TF-IDF chỉ tính từ train. Neural chọn epoch có macro-F1 dev cao nhất, sau đó mới đánh giá test một lần. Không gộp train+dev để fit lại.

Đây là mặc định của template. Các run đã báo cáo phía trên đặt `EPOCHS=25` cho TextCNN và `EPOCHS=20` cho PhoBERT; xem cấu hình và output trong `notebooks_run/` để tái lập đúng các run đó.

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

Các tests kiểm tra cấu trúc và cú pháp ba notebook (outputs trống), preset chung, chính sách bỏ rỗng/giữ nguyên file nguồn, SHA, import package, CSV/mapping/fingerprint bằng câu tự soạn, quy tắc chọn model trên metadata tạm (dev ưu tiên, tie, thiếu model, smoke, dataset/quy tắc lọc khác nhau, điểm không hợp lệ), CLI chỉ ghi cấu hình khi có cờ, kiểm tra đường dẫn và giao diện Streamlit khi thiếu model. Kiểm tra thêm bảng/biểu đồ và cache qua stub UI; đây không phải xác nhận suy luận hoặc đổi artifact thật. Các tests không gọi fit/train, không xuất checkpoint và không để lại metadata giả trong `artifacts/`. Kết quả chạy tests ở `VALIDATION.md`.

**Đã xác nhận thêm ngày 11/11/2026:** nhận và kiểm tra ba artifact full cùng metadata dữ liệu; phiên bản thư viện artifact khớp môi trường local; script chọn PhoBERT và ghi `configs/active_model.json`. Đã dùng Streamlit AppTest với artifact thật để nạp lần lượt TF-IDF → TextCNN → PhoBERT, nhập câu tự soạn, kiểm tra bảng/kết quả và cache đổi model. Các lần kiểm tra này bật `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; sau đó khôi phục cấu hình PhoBERT. Đây là kiểm tra luồng suy luận, không phải đánh giá chất lượng trên bộ dữ liệu mới.

**Giới hạn còn lại:** chưa đánh giá nhiều seed hoặc tập dữ liệu ngoài ViHSD; kiểm tra giao diện tự động không thay thế thử nghiệm người dùng. Luồng tải dataset trực tiếp bằng token HF của từng thành viên cần được kiểm tra với tài khoản có quyền. `VALIDATION.md` lưu biên bản kiểm tra mã ban đầu ngày 10/11/2026; trạng thái có artifact và kết quả thật mới nhất được cập nhật tại README này.

Các đường dẫn bàn giao chính: `notebooks/`, `src/vihsd/`, `scripts/compare_models.py`, `configs/active_model.example.json`, `app.py`. Báo cáo kiểm tra chi tiết ở `VALIDATION.md`.
