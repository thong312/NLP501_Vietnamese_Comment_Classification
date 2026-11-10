"""Validate CSV snapshot and print aggregate metadata, without training or raw comments."""
import argparse
import json
from pathlib import Path
from vihsd.data import load_data

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kiểm tra ViHSD, không train và không in bình luận nguồn.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--config", default=str(ROOT / "configs/dataset.example.json"))
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        _, metadata = load_data(args.data_dir, cfg["text_column"], cfg["label_column"],
                               cfg["label_mapping"], cfg["mapping_source"], cfg["source"],
                               empty_text_policy=cfg["empty_text_policy"],
                               expected_sha256=cfg["expected_sha256"])
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Kiểm tra dữ liệu thất bại: {exc}\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
