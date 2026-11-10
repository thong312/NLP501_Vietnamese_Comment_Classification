"""Contract tests use temporary synthetic metadata, never fitted models or real scores."""
import ast
import csv
import json
from pathlib import Path
import pytest

from vihsd.artifacts import read_manifest, write_json
from vihsd.comparison import collect, main, select
from vihsd.data import load_data
from vihsd.inference import configured_artifact
from vihsd.labels import LABELS
from vihsd.training import require_colab

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks():
    import nbformat
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs and cell.execution_count is None


def dataset():
    return {"source": "synthetic-test-only", "sha256": {s: "a" * 64 for s in ("train", "dev", "test")},
            "used_counts": {s: dict.fromkeys(LABELS, 1) for s in ("train", "dev", "test")}, "seed": 42}


def score(f1):
    return {"labels": LABELS, "macro_f1": f1, "n_samples": 3,
            "per_class": {s: {"precision": f1, "recall": f1, "f1-score": f1, "support": 1} for s in LABELS},
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def row(kind, dev=0.6, test=0.5, size=100, run_type="full", run_id=None):
    return {"manifest": {"model_type": kind, "run_id": run_id or kind, "run_type": run_type,
                         "dataset": dataset(), "labels": LABELS},
            "metrics": {"dev": score(dev), "test": score(test)}, "size": size, "directory": Path(kind)}


def test_selection_dev_only_and_ties():
    rows = [row("tfidf", dev=0.7, test=0.1), row("textcnn", dev=0.6, test=1.0), row("phobert")]
    assert select(rows)[0] is rows[0]
    rows[1]["metrics"]["dev"] = score(0.7)
    rows[1]["size"] = 50
    assert select(rows)[0] is rows[1]
    rows[0]["size"] = 50
    rows[0]["manifest"]["run_id"] = "a"
    rows[1]["manifest"]["run_id"] = "b"
    assert select(rows)[0] is rows[0]


def test_selection_incomplete_smoke_mismatch():
    rows = [row("tfidf"), row("textcnn"), row("phobert", run_type="smoke")]
    assert select(rows)[0] is None
    rows[2]["manifest"]["run_type"] = "full"
    rows[2]["manifest"]["dataset"]["seed"] = 7
    assert select(rows)[0] is None
    assert select([])[0] is None


def write_fake(directory, kind, dev):
    directory.mkdir(parents=True)
    files = {"tfidf": {"pipeline": "pipeline.joblib"},
             "textcnn": {"weights": "weights.safetensors", "vocab": "vocab.json", "architecture": "architecture.json"},
             "phobert": {"model": "model", "tokenizer": "tokenizer"}}[kind]
    for filename in files.values():
        if kind == "phobert":
            target = directory / filename
            target.mkdir()
            names = ("config.json", "model.safetensors") if filename == "model" else ("tokenizer_config.json", "vocab.txt", "bpe.codes")
            for name in names:
                (target / name).write_text("TEST METADATA ONLY; NOT A MODEL", encoding="utf-8")
        else:
            (directory / filename).write_text("TEST METADATA ONLY; NOT A MODEL", encoding="utf-8")
    m = row(kind)["manifest"]
    m.update(schema_version=1, files=files, libraries={}, training={},
             preprocessing={"name": {"tfidf": "NFC+whitespace-v1", "textcnn": "regex-lower-v1", "phobert": "pyvi-v1"}[kind]})
    write_json(directory / "manifest.json", m)
    write_json(directory / "metrics.json", {"schema_version": 1, "run_type": "full", "dev": score(dev), "test": score(0.9)})


def test_compare_cli_write_opt_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for kind, dev in (("tfidf", 0.8), ("textcnn", 0.7), ("phobert", 0.6)):
        write_fake(tmp_path / "artifacts" / kind, kind, dev)
    rows, errors = collect("artifacts")
    assert len(rows) == 3 and not errors
    assert main([]) == 0
    assert not Path("configs/active_model.json").exists()
    main(["--write-config"])
    assert json.loads(Path("configs/active_model.json").read_text())["artifact_dir"] == "artifacts/tfidf"
    metric_path = Path("artifacts/phobert/metrics.json")
    metrics = json.loads(metric_path.read_text())
    metrics["dev"]["macro_f1"] = float("nan")
    write_json(metric_path, metrics)
    rows, errors = collect("artifacts")
    assert len(errors) == 1 and select(rows)[0] is None


def test_artifact_rejects_mapping_and_escape(tmp_path):
    directory = tmp_path / "artifact"
    write_fake(directory, "tfidf", 0.6)
    path = directory / "manifest.json"
    m = json.loads(path.read_text())
    m["labels"] = list(reversed(LABELS))
    write_json(path, m)
    with pytest.raises(ValueError):
        read_manifest(directory)
    m["labels"] = LABELS
    m["files"]["pipeline"] = "../outside"
    (tmp_path / "outside").write_text("test")
    write_json(path, m)
    with pytest.raises(ValueError):
        read_manifest(directory)


def test_data_validation_and_fingerprints(tmp_path):
    for split in ("train", "dev", "test"):
        with (tmp_path / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "code"])
            writer.writerows([(f"Câu tự soạn {i}", str(i)) for i in range(3)])
    args = (tmp_path, "text", "code", {str(i): s for i, s in enumerate(LABELS)}, "synthetic test spec", "synthetic")
    splits, metadata = load_data(*args)
    assert len(splits["train"]) == 3
    assert metadata["cross_split_duplicates"]["train:dev"]["unique_texts"] == 3
    assert len(metadata["sha256"]["train"]) == 64
    with pytest.raises(ValueError):
        load_data(tmp_path, "text", "code", {}, "", "synthetic")
    (tmp_path / "dev.csv").write_text("text,code\n,99\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(*args)


def test_no_local_training():
    with pytest.raises(RuntimeError, match="Colab"):
        require_colab()


def test_missing_and_escaping_config(tmp_path):
    with pytest.raises(ValueError, match="active_model"):
        configured_artifact(tmp_path)
    config = tmp_path / "config.json"
    write_json(config, {"artifact_dir": "../outside"})
    with pytest.raises(ValueError):
        configured_artifact(tmp_path, config)


def test_app_without_model():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not app.exception
    assert "Chưa thể nạp model" in app.warning[0].value
    assert "active_model.json" in app.info[0].value
    assert len(app.text_area) == 0


def test_app_display_with_stub_and_cache_switch(tmp_path, monkeypatch):
    # UI-only stub: no model, fit, checkpoint or experimental result is created.
    import vihsd.inference as inference
    from streamlit.testing.v1 import AppTest
    loaded = []
    class Stub:
        def __init__(self, directory):
            loaded.append(directory)
            self.manifest = {"model_type": "tfidf", "run_id": Path(directory).name, "run_type": "full"}
        def predict(self, text):
            if not text.strip():
                raise ValueError("Vui lòng nhập bình luận có nội dung.")
            return {"label": "CLEAN", "scores": {"CLEAN": 0.8, "OFFENSIVE": 0.1, "HATE": 0.1},
                    "model_id": self.manifest["run_id"]}
    current = [tmp_path / "first"]
    monkeypatch.setattr(inference, "configured_artifact", lambda root: current[0])
    monkeypatch.setattr(inference, "Predictor", Stub)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    app.button[0].click().run()
    assert "nhập bình luận" in app.error[0].value
    app.text_area[0].set_value("Câu tự soạn kiểm tra UI").run()
    app.button[0].click().run()
    assert not app.exception and "CLEAN" in app.subheader[0].value
    assert len(app.table) == 1
    current[0] = tmp_path / "second"
    app.run()
    assert loaded == [str(tmp_path / "first"), str(tmp_path / "second")]
