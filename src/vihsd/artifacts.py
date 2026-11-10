import importlib.metadata
import json
import platform
from pathlib import Path
from .labels import LABELS

MODEL_TYPES = {"tfidf", "textcnn", "phobert"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def versions(model_type):
    packages = ["numpy", "scikit-learn", "joblib"]
    if model_type != "tfidf":
        packages += ["torch", "safetensors"]
    if model_type == "phobert":
        packages += ["transformers", "tokenizers", "pyvi", "sklearn-crfsuite", "python-crfsuite"]
    return {p: importlib.metadata.version(p) for p in packages}


def export_metadata(directory, model_type, run_id, files, preprocessing, dataset, metrics, smoke, training):
    directory = Path(directory)
    manifest = dict(schema_version=1, model_type=model_type, run_id=run_id,
                    labels=LABELS, files=files, preprocessing=preprocessing,
                    libraries=versions(model_type), python=platform.python_version(),
                    dataset=dataset, run_type="smoke" if smoke else "full", training=training)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "metrics.json", dict(schema_version=1, run_type=manifest["run_type"], **metrics))


def artifact_file(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Tên file artifact phải tương đối.")
    base = Path(directory).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.exists():
        raise ValueError(f"Thiếu file hoặc đường dẫn artifact không hợp lệ: {relative}")
    return path


def read_manifest(directory, check_versions=False):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if m.get("schema_version") != 1 or m.get("model_type") not in MODEL_TYPES or m.get("labels") != LABELS:
        raise ValueError("Manifest không tương thích schema/model/mapping nhãn.")
    required = {"run_id", "files", "preprocessing", "libraries", "dataset", "run_type", "training"}
    if not required.issubset(m) or m["run_type"] not in ("full", "smoke") or not m["run_id"]:
        raise ValueError("Manifest thiếu metadata bắt buộc.")
    expected = {"tfidf": {"pipeline"}, "textcnn": {"weights", "vocab", "architecture"},
                "phobert": {"model", "tokenizer"}}[m["model_type"]]
    if set(m["files"]) != expected:
        raise ValueError("Manifest thiếu danh sách file của model.")
    for relative in m["files"].values():
        artifact_file(directory, relative)
    if m["model_type"] == "phobert":
        for key in ("model", "tokenizer"):
            if not artifact_file(directory, m["files"][key]).is_dir():
                raise ValueError(f"PhoBERT cần thư mục {key} từ save_pretrained.")
        artifact_file(directory, m["files"]["model"] + "/config.json")
        artifact_file(directory, m["files"]["model"] + "/model.safetensors")
        for filename in ("tokenizer_config.json", "vocab.txt", "bpe.codes"):
            artifact_file(directory, m["files"]["tokenizer"] + "/" + filename)
    elif any(not artifact_file(directory, p).is_file() for p in m["files"].values()):
        raise ValueError("Artifact cần file suy luận hợp lệ.")
    expected_pre = {"tfidf": "NFC+whitespace-v1", "textcnn": "regex-lower-v1", "phobert": "pyvi-v1"}
    if m["preprocessing"].get("name") != expected_pre[m["model_type"]]:
        raise ValueError("Không hỗ trợ cấu hình tiền xử lý này.")
    if check_versions:
        for package, wanted in m["libraries"].items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise ValueError(f"Thiếu {package}=={wanted}; cài requirements tương ứng.") from None
            if actual != wanted:
                raise ValueError(f"Lệch phiên bản {package}: cần {wanted}, hiện có {actual}. Cài đúng phiên bản Colab.")
    return m
