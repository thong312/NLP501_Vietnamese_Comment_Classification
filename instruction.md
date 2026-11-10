# Triển khai dự án ViHSD cho nhóm 3 người

## Mục tiêu và phạm vi

Bạn là Codex agent phụ trách triển khai dự án **Phân loại bình luận tiếng Việt với ViHSD** từ đầu đến cuối. Hãy tạo code chạy được, giao diện tiếng Việt và README hướng dẫn sử dụng. Đầu ra là một trong ba nhãn: CLEAN (bình thường), OFFENSIVE (xúc phạm), HATE (thù ghét).

Tạo đúng ba notebook .ipynb hoàn chỉnh để mỗi thành viên tự train một model trên Google Colab. **Toàn bộ quá trình fit, train và fine-tune diễn ra trên Colab, kể cả TF-IDF + Logistic Regression.** Local chỉ cài môi trường, kiểm tra code và chạy suy luận với một model đã xuất. Nếu nhóm chưa chạy Colab và chưa cung cấp artifact, ghi rõ trạng thái đó. Không tự tạo điểm số, checkpoint hay model thắng cuộc.

Đọc AGENTS.md và kiểm tra repo trước khi sửa. Giữ cấu trúc sẵn có nếu phù hợp. Với repo mới, dùng Python và Streamlit cho demo local, scikit-learn cho baseline, PyTorch cho TextCNN và Hugging Face Transformers cho PhoBERT. Dùng một app Streamlit duy nhất, tránh thêm frontend và backend riêng nếu không cần.

## Cấu trúc repo đề xuất

```text
README.md
instruction.md
app.py
pyproject.toml
requirements-local.txt
requirements-colab.txt
.gitignore
configs/
  active_model.example.json
notebooks/
  01_tfidf_logistic_regression.ipynb
  02_textcnn.ipynb
  03_phobert.ipynb
src/vihsd/
  labels.py
  data.py
  evaluation.py
  inference.py
  models/
    classical.py
    textcnn.py
    phobert.py
scripts/
  compare_models.py
artifacts/
  .gitkeep
```

Đặt code dùng chung trong package `vihsd` để notebook và app dùng cùng cách tiền xử lý và cùng kiến trúc. Tài liệu hóa lệnh cài package từ root, ví dụ `python -m pip install -e .` sau khi cài dependencies. Đồng bộ phiên bản thư viện cần cho nạp artifact ở Colab và local. Bỏ qua CSV nguồn, artifacts, cấu hình local, secrets và cache trong Git. Giữ file cấu hình mẫu trong Git.

## Dữ liệu ViHSD

- Nguồn ưu tiên: [UIT NLP trên Hugging Face](https://huggingface.co/datasets/uitnlp/vihsd). Bộ dữ liệu có giới hạn truy cập. Hỗ trợ lấy đúng revision bằng token từ Colab Secrets khi thành viên có quyền, hoặc đọc ba CSV mà nhóm đã lấy hợp lệ qua một thư mục cấu hình được. Giữ token ngoài notebook, log và Git. Nếu bị từ chối, báo cách xin quyền hoặc cung cấp CSV. Không tự thay bằng dataset khác hay dùng bản mirror chưa xác minh.
- Dùng split gốc `train.csv`, `dev.csv`, `test.csv`. Xác nhận cột văn bản, cột nhãn và ý nghĩa của mã nhãn dựa trên nguồn trước khi chuyển đổi. Trong code dùng mapping nội bộ duy nhất cho CLEAN, OFFENSIVE, HATE. Nếu mã gốc chưa xác minh được, yêu cầu mapping cấu hình rõ ràng và dừng khi thiếu. Kiểm tra dữ liệu rỗng, nhãn lạ và số mẫu theo lớp. Báo các dòng trùng giữa các split nếu phát hiện và ghi rõ cách xử lý chung.
- Ba notebook dùng cùng snapshot và cùng quy tắc lọc. Ghi nguồn, revision hoặc SHA-256 của từng CSV, số mẫu và seed vào metadata. Vocabulary, TF-IDF và trọng số lớp chỉ lấy từ train. Dùng dev để chọn cấu hình, checkpoint và model. Giữ test cho đánh giá cuối sau khi đã cố định cấu hình. Không dùng điểm test để đổi model.
- Dẫn [repo và bài báo của tác giả](https://github.com/sonlam1102/vihsd) trong README. Giữ demo phục vụ học tập và không commit bình luận nguồn.

## Ba notebook Google Colab

| Notebook | Thành viên train | Artifact suy luận |
|---|---|---|
| 01 | Pipeline TF-IDF + Logistic Regression, chạy CPU | Pipeline đã fit, lưu cả vectorizer và classifier |
| 02 | TextCNN cho phân loại văn bản, dùng PyTorch | Trọng số, vocabulary, PAD/UNK, độ dài tối đa và cấu hình kiến trúc |
| 03 | Fine-tune PhoBERT cho ba nhãn, ưu tiên GPU Colab | Model và tokenizer xuất bằng save_pretrained, cùng cấu hình tiền xử lý |

Mỗi notebook có đầy đủ cell setup, lấy repo ở revision rõ ràng hoặc tải bản repo do nhóm cung cấp, cài thư viện, kiểm tra thiết bị, tải dữ liệu, xem phân bố nhãn, train, đánh giá, dự đoán thử và xuất artifact. Notebook phải chạy theo thứ tự từ một Colab mới, không phụ thuộc đường dẫn máy cá nhân. Có một cell cấu hình dễ sửa và giải thích ngắn phần mà thành viên cần trình bày. Dùng mặc định vừa sức Colab và có chế độ smoke trên tập con nếu cần kiểm tra luồng trên Colab. Kết quả smoke phải được đánh dấu và không được xét khi chọn model.

Ở TextCNN, tái sử dụng cùng tokenizer và cách padding/cắt độ dài khi train và inference. Ở PhoBERT, triển khai tiền xử lý và tách từ tiếng Việt nhất quán với checkpoint đã chọn. Ghi trong README cách cài các phụ thuộc tách từ ở cả Colab và local, nếu có. Sau khi xuất, mỗi notebook phải nạp lại artifact và chạy thử vài câu tự soạn để kiểm tra mapping, tiền xử lý và định dạng đầu ra. Cho phép tải ZIP hoặc lưu ZIP vào Drive của thành viên.

## Hợp đồng artifact và chọn model

Mỗi lần train xuất một thư mục độc lập, ví dụ `artifacts/textcnn/run-001/`. Thư mục chứa `manifest.json`, `metrics.json` và toàn bộ file cần để suy luận. Manifest có version schema, loại model, ID lần train, nhãn theo thứ tự đầu ra, tên file tương đối, cấu hình tiền xử lý, phiên bản thư viện và dấu vết dataset. Metrics ghi riêng dev và test, gồm macro-F1, precision/recall/F1 theo lớp, confusion matrix có thứ tự nhãn cố định và số mẫu. Ghi trạng thái full hoặc smoke. Artifact PhoBERT cần tự chứa model/tokenizer để suy luận local không cần mạng sau khi cài đủ phụ thuộc.

Viết `scripts/compare_models.py` để đọc kết quả của ba notebook, kiểm tra cùng dataset, split, thứ tự nhãn và loại run, rồi in bảng so sánh. Chọn mặc định theo **macro-F1 dev cao nhất**. Khi bằng điểm, ưu tiên model có artifact nhỏ hơn rồi dùng ID làm tiêu chí ổn định. Hiển thị F1 từng lớp và điểm test riêng để báo cáo. Khi chưa đủ ba kết quả hợp lệ, hiển thị trạng thái tạm thời và không công bố model tốt nhất. Script chỉ cập nhật cấu hình nếu người dùng truyền cờ rõ ràng, ví dụ `--write-config`.

## Giao diện và đổi model local

- `app.py` đọc `configs/active_model.json`. Cấu hình chỉ trỏ đến một thư mục artifact. Bộ nạp đọc manifest, chọn adapter tương ứng, kiểm tra file và mapping nhãn, rồi nạp đúng model đó trên CPU. Chỉ import backend model khi cần. Các adapter trả về cùng cấu trúc gồm nhãn, điểm dự đoán theo từng lớp và ID model.
- Giao diện có ô nhập bình luận, nút phân loại, nhãn kết quả, điểm từng lớp và tên model đang dùng. Xử lý đầu vào rỗng và giới hạn độ dài hợp lý. Nếu chưa có cấu hình, artifact thiếu hoặc không tương thích, hiển thị hướng dẫn sửa cụ thể. Gọi điểm đầu ra là điểm dự đoán của model, không diễn giải thành xác suất đã hiệu chỉnh.
- Đổi model bằng cách đặt thư mục artifact khác vào cấu hình rồi khởi động lại app. Không sửa app để đổi model và không nạp đồng thời ba model. Dùng đường dẫn tương đối tính từ root repo, kiểm tra cache không giữ model cũ khi chuyển.
- Dùng artifact từ ba notebook của nhóm. Với định dạng có thể thực thi mã khi nạp như joblib, không hỗ trợ upload model qua giao diện hoặc nạp artifact không rõ nguồn. Dùng safetensors hoặc định dạng chỉ chứa trọng số cho model PyTorch nếu phù hợp.

## README và nghiệm thu

README tiếng Việt phải có lệnh cài đặt, phiên bản Python đã kiểm tra, sơ đồ thư mục, phân công ba thành viên, hướng dẫn lấy ViHSD, cách mở và Run all từng notebook trên Colab, cách xuất và giải nén artifact, cách so sánh rồi chọn model, ví dụ cấu hình đổi model và lệnh `streamlit run app.py`. Viết lệnh thật, đồng bộ với CLI và code. Giải thích cách khắc phục thiếu quyền dữ liệu, hết GPU, thiếu artifact hoặc lệch phiên bản. Nếu chưa có kết quả, trình bày bảng ở trạng thái chờ kết quả Colab.

Trước khi bàn giao, kiểm tra cấu trúc và cú pháp cả ba notebook, import package, quy tắc chọn model bằng metadata giả lập, và giao diện khi thiếu model. **Không fit hoặc train local để kiểm tra.** Nếu có artifact thật do nhóm cung cấp, chạy smoke inference local và kiểm tra đổi cấu hình sang một artifact khác nếu có. Ghi rõ những bước đã kiểm tra, những bước chỉ có thể xác nhận sau khi nhóm chạy Colab, và các đường dẫn bàn giao. Chỉ báo đã hoàn thành huấn luyện hoặc đã xác nhận đổi model thực tế khi có kết quả tương ứng.
