import json
import math
from pathlib import Path
from .artifacts import MODEL_TYPES, read_manifest, write_json
from .labels import LABELS


def validate_metric(metric, expected_count):
    if metric.get("labels") != LABELS or metric.get("n_samples") != expected_count:
        raise ValueError("Metrics sai thứ tự nhãn hoặc số mẫu.")
    values = [metric["macro_f1"]]
    if set(metric["per_class"]) != set(LABELS):
        raise ValueError("Metrics thiếu lớp.")
    for label in LABELS:
        values.extend(metric["per_class"][label][k] for k in ("precision", "recall", "f1-score"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Metrics có điểm ngoài [0, 1].")
    cm = metric["confusion_matrix"]
    if len(cm) != 3 or any(len(row) != 3 for row in cm):
        raise ValueError("Confusion matrix phải có kích thước 3x3.")
    if any(type(v) is not int or v < 0 for row in cm for v in row) or sum(map(sum, cm)) != expected_count:
        raise ValueError("Confusion matrix không khớp số mẫu.")
    if abs(metric["macro_f1"] - sum(metric["per_class"][s]["f1-score"] for s in LABELS) / 3) > 1e-8:
        raise ValueError("Macro-F1 không khớp F1 từng lớp.")


def collect(root):
    rows, errors = [], []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        directory = manifest_path.parent
        try:
            m = read_manifest(directory)
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if metrics.get("schema_version") != 1 or metrics.get("run_type") != m["run_type"]:
                raise ValueError("Metrics và manifest khác schema/run_type.")
            dataset = m["dataset"]
            if set(dataset.get("sha256", {})) != {"train", "dev", "test"}:
                raise ValueError("Thiếu fingerprint các split.")
            for split in ("dev", "test"):
                count = sum(dataset["used_counts"][split].values())
                validate_metric(metrics[split], count)
                if any(metrics[split]["per_class"][label]["support"] != dataset["used_counts"][split][label] for label in LABELS):
                    raise ValueError("Support không khớp phân bố lớp.")
            size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            rows.append({"directory": directory, "manifest": m, "metrics": metrics, "size": size})
        except (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError) as exc:
            errors.append(f"{directory}: {exc}")
    return rows, errors


def select(rows):
    full = [r for r in rows if r["manifest"]["run_type"] == "full"]
    if {r["manifest"]["model_type"] for r in full} != MODEL_TYPES:
        return None, "Tạm thời: cần đủ ba loại model có kết quả full hợp lệ; smoke không được xét."
    fingerprints = {json.dumps(r["manifest"]["dataset"], sort_keys=True) for r in full}
    if len(fingerprints) != 1:
        return None, "Tạm thời: dataset/split/mapping/seed/quy tắc lọc khác nhau; chưa thể chọn model."
    best = min(full, key=lambda r: (-r["metrics"]["dev"]["macro_f1"], r["size"],
                                    r["manifest"]["run_id"], str(r["directory"])))
    return best, "Đã đủ ba loại model full trên cùng dữ liệu; chọn theo macro-F1 dev."


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="So sánh kết quả Colab ViHSD; test chỉ để báo cáo.")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--config", default="configs/active_model.json")
    args = parser.parse_args(argv)
    rows, errors = collect(args.artifacts)
    print("Model | Run | Loại run | Dev macro-F1 | Dev F1 CLEAN/OFFENSIVE/HATE | Test macro-F1 | Test F1 CLEAN/OFFENSIVE/HATE | Bytes")
    for row in rows:
        m, metric = row["manifest"], row["metrics"]
        per = lambda split: "/".join(f"{metric[split]['per_class'][s]['f1-score']:.4f}" for s in LABELS)
        print(f"{m['model_type']} | {m['run_id']} | {m['run_type']} | {metric['dev']['macro_f1']:.4f} | "
              f"{per('dev')} | {metric['test']['macro_f1']:.4f} | {per('test')} | {row['size']}")
    for error in errors:
        print("Bỏ qua artifact không hợp lệ:", error)
    winner, status = select(rows)
    print(status)
    if winner:
        print("Model được chọn:", winner["manifest"]["model_type"], winner["manifest"]["run_id"])
        if args.write_config:
            path = winner["directory"].resolve()
            root = Path.cwd().resolve()
            if not path.is_relative_to(root):
                parser.error("Artifact cần nằm trong root repo để ghi cấu hình.")
            config = Path(args.config)
            config.parent.mkdir(parents=True, exist_ok=True)
            write_json(config, {"artifact_dir": path.relative_to(root).as_posix()})
            print("Đã ghi:", config)
    elif args.write_config:
        print("Không ghi cấu hình khi chưa có model được chọn.")
    return 0
