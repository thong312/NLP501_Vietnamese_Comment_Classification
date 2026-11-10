"""Strict CSV loading; no implicit guesses about source label codes."""
import csv
import hashlib
import random
import re
import unicodedata
from pathlib import Path

from .labels import LABELS, LABEL_TO_ID


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def load_data(directory, text_column, label_column, label_mapping, mapping_source,
              source, revision=None, seed=42, smoke=False):
    if not mapping_source or not label_mapping or set(label_mapping.values()) != set(LABELS):
        raise ValueError("Cần mapping mã gốc → CLEAN/OFFENSIVE/HATE và nguồn xác minh mapping.")
    if not text_column or not label_column or not source:
        raise ValueError("Cần khai báo cột văn bản, cột nhãn và nguồn dữ liệu.")
    mapping = {str(k): v for k, v in label_mapping.items()}
    splits, fingerprints, counts = {}, {}, {}
    for split in ("train", "dev", "test"):
        path = Path(directory) / f"{split}.csv"
        fingerprints[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {text_column, label_column}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name}: không tìm thấy cột đã cấu hình.")
            rows = []
            for line, row in enumerate(reader, 2):
                text, code = normalize(row[text_column] or ""), row[label_column]
                if not text or code not in mapping:
                    raise ValueError(f"{path.name}, dòng {line}: văn bản rỗng hoặc nhãn lạ.")
                rows.append((text, LABEL_TO_ID[mapping[code]]))
        counts[split] = {label: sum(y == i for _, y in rows) for i, label in enumerate(LABELS)}
        if not rows or any(n == 0 for n in counts[split].values()):
            raise ValueError(f"{split}: rỗng hoặc thiếu lớp: {counts[split]}")
        splits[split] = rows
    overlaps = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = {t for t, _ in splits[a]} & {t for t, _ in splits[b]}
        overlaps[f"{a}:{b}"] = {
            "unique_texts": len(common),
            "rows": {s: sum(t in common for t, _ in splits[s]) for s in (a, b)},
        }
    # Preserve official splits, including duplicates; report counts without logging raw text.
    metadata = dict(source=source, revision=revision, sha256=fingerprints,
                    columns={"text": text_column, "label": label_column},
                    label_mapping=mapping, mapping_source=mapping_source,
                    normalization="NFC+whitespace-v1", duplicate_policy="report-and-preserve-original-splits",
                    cross_split_duplicates=overlaps, original_counts=counts, seed=seed)
    if smoke:
        for split, rows in splits.items():
            rng = random.Random(seed)
            selected = []
            for i in range(3):
                group = [r for r in rows if r[1] == i]
                rng.shuffle(group)
                selected.extend(group[:32])
            rng.shuffle(selected)
            splits[split] = selected
    metadata["used_counts"] = {s: {label: sum(y == i for _, y in rows)
                                    for i, label in enumerate(LABELS)} for s, rows in splits.items()}
    return splits, metadata


def download_csvs(revision, filenames, token):
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("HF_REVISION phải là commit SHA đầy đủ (40 ký tự).")
    from huggingface_hub import hf_hub_download
    if set(filenames) != {"train", "dev", "test"}:
        raise ValueError("Cần đường dẫn ba file CSV chính thức.")
    import shutil
    target = Path("/content/vihsd-data")
    target.mkdir(exist_ok=True)
    try:
        for split, filename in filenames.items():
            cached = hf_hub_download("uitnlp/vihsd", filename, repo_type="dataset",
                                     revision=revision, token=token)
            shutil.copyfile(cached, target / f"{split}.csv")
    except Exception:
        raise RuntimeError("Không tải được ViHSD. Xin quyền tại uitnlp/vihsd, kiểm tra SHA/tên file, "
                           "HF_TOKEN trong Colab Secrets hoặc dùng CSV được cấp hợp lệ.") from None
    return target
