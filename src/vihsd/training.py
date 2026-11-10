"""Training entry points are deliberately restricted to Google Colab."""
import random
from pathlib import Path

from .artifacts import export_metadata, write_json
from .evaluation import evaluate
from .labels import LABELS


def require_colab():
    try:
        import google.colab  # noqa: F401
    except ImportError:
        raise RuntimeError("Huấn luyện chỉ được phép trên Google Colab; local chỉ suy luận/kiểm tra.") from None


def seed_everything(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xy(rows):
    return [t for t, _ in rows], [y for _, y in rows]


def train_tfidf(splits, directory, seed):
    require_colab()
    import joblib
    from .models.classical import make_pipeline
    train_x, train_y = xy(splits["train"])
    dev_x, dev_y = xy(splits["dev"])
    best, best_score, selected_c = None, -1, None
    for c in (0.5, 1.0, 2.0):
        model = make_pipeline(c)
        model.set_params(classifier__random_state=seed)
        model.fit(train_x, train_y)
        score = evaluate(dev_y, model.predict(dev_x))["macro_f1"]
        print(f"C={c}; dev macro-F1={score:.4f}")
        if score > best_score:
            best, best_score, selected_c = model, score, c
    joblib.dump(best, directory / "pipeline.joblib")
    metrics = {s: evaluate(xy(splits[s])[1], best.predict(xy(splits[s])[0])) for s in ("dev", "test")}
    return {"pipeline": "pipeline.joblib"}, {"name": "NFC+whitespace-v1"}, metrics, {"selected_C": selected_c}


def train_neural(kind, splits, directory, seed, epochs, batch_size, model_revision=None):
    require_colab()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.utils.class_weight import compute_class_weight
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)
    if kind == "phobert" and device.type != "cuda":
        print("Cảnh báo: PhoBERT trên CPU rất chậm. Nên bật GPU Colab.")
    train_text, train_y = xy(splits["train"])
    max_length = 128
    if kind == "textcnn":
        from .models.textcnn import TextCNN, build_vocab, encode
        vocab = build_vocab(train_text)
        arch = dict(vocab_size=len(vocab), embedding_dim=128, channels=100,
                    kernels=[3, 4, 5], dropout=0.5)
        model = TextCNN(**arch)
        tensors = {s: (torch.tensor([encode(t, vocab, max_length) for t, _ in rows]),
                       torch.tensor([y for _, y in rows])) for s, rows in splits.items()}
        def collate(rows):
            return {"input_ids": torch.stack([r[0] for r in rows])}, torch.stack([r[1] for r in rows])
        datasets = {s: list(zip(*values)) for s, values in tensors.items()}
        learning_rate = 1e-3
    else:
        import re
        if not model_revision or not re.fullmatch(r"[a-fA-F0-9]{40}", model_revision):
            raise ValueError("PHOBERT_REVISION phải là SHA commit đầy đủ của vinai/phobert-base-v2.")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from .models.phobert import segment
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", revision=model_revision, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base-v2", revision=model_revision, num_labels=3,
            id2label=dict(enumerate(LABELS)), label2id={s: i for i, s in enumerate(LABELS)})
        datasets = {s: [(segment(t), y) for t, y in rows] for s, rows in splits.items()}
        def collate(rows):
            inputs = tokenizer([r[0] for r in rows], truncation=True, padding=True,
                               max_length=max_length, return_tensors="pt")
            return dict(inputs), torch.tensor([r[1] for r in rows])
        learning_rate = 2e-5
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(rows, batch_size=batch_size, shuffle=(s == "train"),
                             collate_fn=collate, generator=generator, num_workers=0)
               for s, rows in datasets.items()}
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(train_y))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def logits(inputs):
        output = model(**{key: value.to(device) for key, value in inputs.items()})
        return output if kind == "textcnn" else output.logits

    def score(split):
        model.eval()
        truth, predictions = [], []
        with torch.inference_mode():
            for inputs, target in loaders[split]:
                predictions.extend(logits(inputs).argmax(-1).cpu().tolist())
                truth.extend(target.tolist())
        return evaluate(truth, predictions)

    best_score, best_state, best_epoch = -1, None, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits(inputs), target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        dev = score("dev")
        print(f"Epoch {epoch + 1}: loss={total_loss / len(loaders['train']):.4f}; dev macro-F1={dev['macro_f1']:.4f}")
        if dev["macro_f1"] > best_score:
            best_score, best_epoch = dev["macro_f1"], epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Test is evaluated only once, after checkpoint selection on dev is fixed.
    metrics = {s: score(s) for s in ("dev", "test")}
    model.cpu().eval()
    if kind == "textcnn":
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(directory / "weights.safetensors"))
        write_json(directory / "vocab.json", vocab)
        write_json(directory / "architecture.json", arch)
        files = {"weights": "weights.safetensors", "vocab": "vocab.json", "architecture": "architecture.json"}
        preprocessing = {"name": "regex-lower-v1", "max_length": max_length, "PAD": 0, "UNK": 1}
    else:
        model.save_pretrained(directory / "model", safe_serialization=True)
        tokenizer.save_pretrained(directory / "tokenizer")
        files = {"model": "model", "tokenizer": "tokenizer"}
        preprocessing = {"name": "pyvi-v1", "max_length": max_length, "segmenter": "pyvi==0.1.1"}
    training = {"best_epoch": best_epoch, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "class_weights": weights.tolist(),
                "pretrained_revision": model_revision if kind == "phobert" else None,
                "device": str(device), "selection": "dev_macro_f1"}
    return files, preprocessing, metrics, training


def run_training(kind, splits, dataset, run_id, smoke=False, epochs=None, batch_size=None, model_revision=None,
                 code_provenance=None):
    require_colab()
    if kind not in ("tfidf", "textcnn", "phobert"):
        raise ValueError("Loại model không hợp lệ.")
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id phải là tên thư mục đơn.")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs phải >= 1.")
    directory = Path("artifacts") / kind / run_id
    directory.mkdir(parents=True, exist_ok=False)
    if kind == "tfidf":
        result = train_tfidf(splits, directory, dataset["seed"])
    else:
        result = train_neural(kind, splits, directory, dataset["seed"],
                              epochs or (1 if smoke else 5 if kind == "textcnn" else 3),
                              batch_size or (64 if kind == "textcnn" else 8), model_revision)
    files, preprocessing, metrics, training = result
    training["code_provenance"] = code_provenance
    export_metadata(directory, kind, run_id, files, preprocessing, dataset, metrics, smoke, training)
    return directory
