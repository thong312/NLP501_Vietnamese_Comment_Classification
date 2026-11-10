"""Rebuild the three source-controlled, output-free Colab notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = json.loads((ROOT / "configs/dataset.example.json").read_text(encoding="utf-8"))


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CONFIG = '''
# Điền giống nhau ở cả ba notebook trước khi Runtime → Run all.
SOURCE_MODE = "zip"  # "zip": upload ZIP repo nhóm; "git": clone URL + SHA
REPO_URL = ""
REPO_REVISION = ""  # SHA commit đầy đủ, không dùng main
DATA_MODE = "csv"  # "csv": upload train.csv/dev.csv/test.csv; "hf": Colab Secret HF_TOKEN
CSV_DIR = "/content/vihsd-data"  # có thể đổi sang thư mục Drive đã mount
UPLOAD_CSV = True  # False nếu CSV_DIR đã chứa đủ ba CSV hợp lệ
HF_REVISION = ""  # SHA commit đầy đủ của uitnlp/vihsd nếu DATA_MODE="hf"
HF_FILENAMES = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
# Preset cho đúng ba CSV nhóm đã tải và kiểm tra; không chứa bình luận nguồn.
DATASET_CONFIG = __DATASET_CONFIG__
TEXT_COLUMN = DATASET_CONFIG["text_column"]
LABEL_COLUMN = DATASET_CONFIG["label_column"]
LABEL_MAPPING = DATASET_CONFIG["label_mapping"]
MAPPING_SOURCE = DATASET_CONFIG["mapping_source"]
CSV_SOURCE = DATASET_CONFIG["source"]
EMPTY_TEXT_POLICY = DATASET_CONFIG["empty_text_policy"]  # drop: ghi nhận và bỏ văn bản rỗng
EXPECTED_SHA256 = DATASET_CONFIG["expected_sha256"]  # cùng snapshot cho cả ba notebook
SEED = 42
SMOKE = True  # đổi False cho lần train chính thức; smoke không được chọn model
RUN_ID = "{kind}-smoke-001"  # dùng ID mới cho mỗi lần chạy; không ghi đè artifact
PHOBERT_REVISION = ""  # SHA commit đầy đủ vinai/phobert-base-v2, chỉ notebook 03 cần
EPOCHS = None  # mặc định smoke=1; full TextCNN=5, PhoBERT=3
BATCH_SIZE = None  # TextCNN=64, PhoBERT=8; giảm còn 4/2 nếu hết VRAM
DOWNLOAD_ZIP = True
SAVE_TO_DRIVE = False
DRIVE_OUTPUT = "/content/drive/MyDrive/ViHSD"
'''

SETUP = '''
import os, sys, subprocess, pathlib, hashlib, zipfile, shutil, re
try:
    from google.colab import files
except ImportError:
    raise RuntimeError("Notebook này chỉ train trên Google Colab.") from None
workspace = pathlib.Path("/content/vihsd-project")
if SOURCE_MODE == "git":
    if not REPO_URL or not re.fullmatch(r"[a-fA-F0-9]{40}", REPO_REVISION):
        raise ValueError("Điền REPO_URL và REPO_REVISION là SHA commit 40 ký tự.")
    if not workspace.exists():
        subprocess.run(["git", "clone", REPO_URL, str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", "--detach", REPO_REVISION], check=True)
    actual_sha = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
    if actual_sha.lower() != REPO_REVISION.lower():
        raise ValueError("SHA repo không khớp.")
    CODE_PROVENANCE = {"mode": "git", "revision": actual_sha}
elif SOURCE_MODE == "zip":
    print("Upload ZIP mã nguồn của nhóm (không chứa dataset, token hay artifact).")
    uploaded = files.upload()
    if len(uploaded) != 1 or not next(iter(uploaded)).lower().endswith(".zip"):
        raise ValueError("Cần đúng một ZIP repo.")
    name, payload = next(iter(uploaded.items()))
    CODE_PROVENANCE = {"mode": "zip", "sha256": hashlib.sha256(payload).hexdigest()}
    archive_path = pathlib.Path("/content") / pathlib.Path(name).name
    archive_path.write_bytes(payload)
    # Fresh source folder for every upload; reject paths escaping its root.
    import tempfile
    unpack = pathlib.Path(tempfile.mkdtemp(prefix="vihsd-source-", dir="/content"))
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (unpack / member.filename).resolve().is_relative_to(unpack.resolve()):
                raise ValueError("ZIP chứa đường dẫn không hợp lệ.")
        archive.extractall(unpack)
    candidates = [p.parent for p in unpack.rglob("pyproject.toml") if (p.parent / "src/vihsd").is_dir()]
    if len(candidates) != 1:
        raise ValueError("ZIP phải chứa đúng một root dự án có pyproject.toml và src/vihsd.")
    workspace = candidates[0]
else:
    raise ValueError("SOURCE_MODE chỉ nhận zip hoặc git.")
os.chdir(workspace)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", "."], check=True)
# sys.path also supports editable install in the already-running notebook interpreter.
sys.path.insert(0, str(workspace / "src"))
import torch
print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Nguồn code:", CODE_PROVENANCE)
'''

DATA = '''
from vihsd.data import load_data, download_csvs
if DATA_MODE == "hf":
    from google.colab import userdata
    try:
        token = userdata.get("HF_TOKEN")
    except Exception:
        raise RuntimeError("Thêm HF_TOKEN vào Colab Secrets, cấp quyền notebook và xin quyền uitnlp/vihsd.") from None
    try:
        data_dir = download_csvs(HF_REVISION, HF_FILENAMES, token)
    finally:
        del token
    source, revision = "https://huggingface.co/datasets/uitnlp/vihsd", HF_REVISION
elif DATA_MODE == "csv":
    if SAVE_TO_DRIVE or str(CSV_DIR).startswith("/content/drive/"):
        from google.colab import drive
        drive.mount("/content/drive")
    data_dir = pathlib.Path(CSV_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    if UPLOAD_CSV:
        print("Upload đồng thời train.csv, dev.csv và test.csv đã được cấp hợp lệ.")
        uploaded_data = files.upload()
        if set(uploaded_data) != {"train.csv", "dev.csv", "test.csv"}:
            raise ValueError("Cần đúng ba file train.csv, dev.csv, test.csv.")
        for name, payload in uploaded_data.items():
            (data_dir / name).write_bytes(payload)
        del uploaded_data
    source, revision = CSV_SOURCE, None
else:
    raise ValueError("DATA_MODE chỉ nhận csv hoặc hf.")
splits, dataset_metadata = load_data(data_dir, TEXT_COLUMN, LABEL_COLUMN, LABEL_MAPPING,
                                   MAPPING_SOURCE, source, revision, SEED, SMOKE,
                                   empty_text_policy=EMPTY_TEXT_POLICY,
                                   expected_sha256=EXPECTED_SHA256)
print("Phân bố nhãn gốc:", dataset_metadata["original_counts"])
print("Bỏ văn bản rỗng theo quy tắc chung:", dataset_metadata["dropped_empty_text"])
print("Phân bố sau lọc:", dataset_metadata["filtered_counts"])
print("Số mẫu dùng:", dataset_metadata["used_counts"])
print("Trùng giữa split (giữ nguyên split gốc, không in bình luận):", dataset_metadata["cross_split_duplicates"])
print("SHA-256:", dataset_metadata["sha256"])
import pandas as pd
pd.DataFrame(dataset_metadata["used_counts"]).plot.bar(title="Phân bố nhãn theo split")
'''

TRAIN = '''
from vihsd.training import run_training
artifact_dir = run_training("{kind}", splits, dataset_metadata, RUN_ID, smoke=SMOKE,
                            epochs=EPOCHS, batch_size=BATCH_SIZE,
                            model_revision=PHOBERT_REVISION, code_provenance=CODE_PROVENANCE)
print("Đã xuất artifact:", artifact_dir)
'''

REPORT = '''
import json
metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
print("Trạng thái:", metrics["run_type"])
for split in ("dev", "test"):
    print(split, "macro-F1:", metrics[split]["macro_f1"])
    display(pd.DataFrame(metrics[split]["per_class"]).T)
    display(pd.DataFrame(metrics[split]["confusion_matrix"],
                         index=["Thật CLEAN", "Thật OFFENSIVE", "Thật HATE"],
                         columns=["Đoán CLEAN", "Đoán OFFENSIVE", "Đoán HATE"]))
print("Test chỉ phục vụ báo cáo cuối, không đổi cấu hình/model theo điểm test.")
'''

RELOAD = '''
from vihsd.inference import Predictor
reloaded = Predictor(artifact_dir)  # CPU; chỉ đọc file đã xuất, PhoBERT local_files_only=True
for sentence in ["Cảm ơn bạn đã chia sẻ thông tin.", "Mình không đồng ý với nhận xét này.",
                 "Bài viết hôm nay rất hữu ích!"]:
    result = reloaded.predict(sentence)
    assert set(result["scores"]) == {"CLEAN", "OFFENSIVE", "HATE"}
    assert result["model_id"] == RUN_ID
    assert result["label"] in result["scores"]
    print(result)
del reloaded
'''

ZIP = '''
# Archive root contains manifest.json; unzip it into artifacts/<model>/<run-id>/ locally.
zip_path = shutil.make_archive(str(artifact_dir.resolve()), "zip", root_dir=artifact_dir)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    pathlib.Path(DRIVE_OUTPUT).mkdir(parents=True, exist_ok=True)
    destination = pathlib.Path(DRIVE_OUTPUT) / pathlib.Path(zip_path).name
    shutil.copyfile(zip_path, destination)
    print("Đã lưu ZIP vào Drive:", destination)
if DOWNLOAD_ZIP:
    files.download(zip_path)
print("Không commit ZIP/artifact/CSV/token vào Git.")
'''

SPECS = [
    ("01_tfidf_logistic_regression", "tfidf", "TF-IDF + Logistic Regression",
     "Thành viên 1: giải thích TF-IDF unigram/bigram, trọng số lớp chỉ từ train và chọn C trên dev. CPU đủ dùng. Pipeline xuất bao gồm vectorizer và classifier đã fit."),
    ("02_textcnn", "textcnn", "TextCNN",
     "Thành viên 2: trình bày tokenizer regex, vocabulary chỉ từ train, PAD=0/UNK=1, padding/cắt 128 token, embedding, convolution 3/4/5 và max pooling. Chọn epoch theo dev macro-F1; GPU được khuyến nghị."),
    ("03_phobert", "phobert", "PhoBERT",
     "Thành viên 3: bật T4 GPU, điền SHA checkpoint vinai/phobert-base-v2. Trình bày tách từ PyVi nhất quán train/inference, fine-tune và chọn checkpoint theo dev. PyVi là lựa chọn gọn có tài nguyên offline; khác RDRSegmenter được VinAI khuyến nghị nên cần ghi rõ trong báo cáo."),
]

for filename, kind, title, explanation in SPECS:
    cells = [
        cell("markdown", f"# {title} trên Google Colab\n\n{explanation}\n\n"
             "Chỉ train trên Colab. Sửa cấu hình, rồi **Runtime → Run all**. Lần đầu giữ SMOKE=True để kiểm tra luồng; "
             "sau đó dùng ID mới và SMOKE=False cho kết quả chính thức. Cả nhóm phải dùng cùng ZIP/SHA repo và cùng dữ liệu/cấu hình mapping/seed. "
             "Chưa có kết quả cho tới khi notebook chạy thành công. Nếu Colab yêu cầu restart sau pip, restart rồi Run all."),
        cell("markdown", "## 1. Cấu hình\n\nĐã điền preset cho CSV của nhóm: free_text/label_id; 0=CLEAN, 1=OFFENSIVE, 2=HATE, đối chiếu notebook tác giả. "
             "SHA-256 cố định cùng snapshot; bỏ dòng văn bản rỗng có ghi metadata. Không cần đổi tên dev.csv. "
             "Không điền token tại đây: dùng Colab Secrets với tên HF_TOKEN. Với HF, xác nhận đường dẫn file từ Files của revision được cấp quyền."),
        cell("code", CONFIG.replace("{kind}", kind).replace("__DATASET_CONFIG__", repr(DATASET_CONFIG))),
        cell("markdown", "## 2. Lấy source có dấu vết và cài môi trường\n\nZIP chỉ chứa source của nhóm; chia sẻ cùng một ZIP cho ba thành viên. "
             "Nếu clone Git, dùng URL repo nhóm đã đưa source lên và SHA commit đầy đủ."),
        cell("code", SETUP),
        cell("markdown", "## 3. Dữ liệu và phân bố nhãn\n\nDùng split gốc, giữ nguyên dòng trùng và báo số lượng. "
             "Bỏ văn bản rỗng theo preset chung, ghi số lượng và vị trí bản ghi (header tính là bản ghi 1); "
             "dừng nếu SHA khác, nhãn lạ, toàn bộ split rỗng hoặc thiếu lớp sau lọc. Không thay dataset."),
        cell("code", DATA),
        cell("markdown", "## 4. Huấn luyện và cố định checkpoint\n\nFit/vocabulary/trọng số lớp chỉ trên train. Dev chọn cấu hình/checkpoint; test chỉ đánh giá cuối."),
        cell("code", TRAIN.replace("{kind}", kind)),
        cell("markdown", "## 5. Báo cáo dev và test riêng\n\nConfusion matrix có hàng là nhãn thật, cột là nhãn dự đoán; thứ tự CLEAN/OFFENSIVE/HATE."),
        cell("code", REPORT),
        cell("markdown", "## 6. Nạp lại artifact và dự đoán thử\n\nCâu tự soạn chỉ kiểm tra luồng; không dùng làm số liệu đánh giá."),
        cell("code", RELOAD),
        cell("markdown", "## 7. Xuất ZIP\n\nTải về hoặc lưu Drive, rồi dùng script compare_models.py tại local khi có đủ kết quả ba model."),
        cell("code", ZIP),
    ]
    notebook = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "colab": {"name": filename + ".ipynb"}},
                    nbformat=4, nbformat_minor=5)
    for i, c in enumerate(cells):
        c["id"] = f"{kind}-{i:02d}"
    target = ROOT / "notebooks" / (filename + ".ipynb")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Đã tạo đúng ba notebook.")
