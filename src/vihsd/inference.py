import json
from pathlib import Path
from .artifacts import artifact_file, read_manifest
from .data import normalize
from .labels import LABELS


def configured_artifact(root, config_path=None):
    root = Path(root).resolve()
    config = Path(config_path) if config_path else root / "configs/active_model.json"
    if not config.exists():
        raise ValueError("Chưa có configs/active_model.json. Sao chép active_model.example.json, "
                         "sửa artifact_dir sau khi giải nén kết quả Colab.")
    obj = json.loads(config.read_text(encoding="utf-8"))
    if set(obj) != {"artifact_dir"} or not isinstance(obj["artifact_dir"], str):
        raise ValueError("Cấu hình chỉ được chứa artifact_dir.")
    rel = Path(obj["artifact_dir"])
    result = (root / rel).resolve()
    if rel.is_absolute() or not result.is_relative_to(root):
        raise ValueError("artifact_dir phải là đường dẫn tương đối trong root repo.")
    return result


class Predictor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = m = read_manifest(directory, check_versions=True)
        files = {k: artifact_file(directory, v) for k, v in m["files"].items()}
        self.kind = m["model_type"]
        if self.kind == "tfidf":
            import joblib
            self.model = joblib.load(files["pipeline"])
            if list(self.model.classes_) != [0, 1, 2]:
                raise ValueError("Pipeline có thứ tự lớp không hợp lệ.")
        elif self.kind == "textcnn":
            from .models.textcnn import TextCNN
            from safetensors.torch import load_file
            self.vocab = json.loads(files["vocab"].read_text(encoding="utf-8"))
            arch = json.loads(files["architecture"].read_text(encoding="utf-8"))
            if self.vocab.get("<PAD>") != 0 or self.vocab.get("<UNK>") != 1:
                raise ValueError("Sai PAD/UNK.")
            if sorted(self.vocab.values()) != list(range(len(self.vocab))):
                raise ValueError("Vocabulary không liên tục.")
            self.model = TextCNN(**arch)
            self.model.load_state_dict(load_file(str(files["weights"]), device="cpu"))
            self.model.eval()
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(files["tokenizer"], local_files_only=True, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                files["model"], local_files_only=True, use_safetensors=True).cpu().eval()
            if [self.model.config.id2label.get(i) for i in range(3)] != LABELS:
                raise ValueError("PhoBERT có mapping nhãn không khớp manifest.")

    def predict(self, text):
        if not isinstance(text, str) or not normalize(text):
            raise ValueError("Vui lòng nhập bình luận có nội dung.")
        if len(text) > 5000:
            raise ValueError("Bình luận tối đa 5.000 ký tự.")
        if self.kind == "tfidf":
            scores = self.model.predict_proba([text])[0].tolist()
        else:
            import torch
            with torch.inference_mode():
                length = self.manifest["preprocessing"]["max_length"]
                if self.kind == "textcnn":
                    from .models.textcnn import encode
                    logits = self.model(torch.tensor([encode(text, self.vocab, length)]))
                else:
                    from .models.phobert import segment
                    inputs = self.tokenizer(segment(text), truncation=True, max_length=length, return_tensors="pt")
                    logits = self.model(**inputs).logits
                scores = logits.softmax(-1)[0].tolist()
        import math
        if len(scores) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in scores):
            raise ValueError("Model trả về điểm không hợp lệ.")
        return {"label": LABELS[max(range(3), key=lambda i: scores[i])],
                "scores": dict(zip(LABELS, scores)), "model_id": self.manifest["run_id"]}
