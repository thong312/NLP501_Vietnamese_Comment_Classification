"""Package only project source, preserving src/vihsd and excluding local state."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    candidates = [ROOT / name for name in ("app.py", "pyproject.toml", "README.md", "VALIDATION.md", "DATA_REPORT.md", "instruction.md", ".gitignore")]
    candidates += list(ROOT.glob("requirements-*.txt"))
    candidates += list((ROOT / "configs").glob("*.example.json"))
    candidates += [ROOT / "artifacts/.gitkeep"]
    for folder in ("src/vihsd", "scripts", "notebooks", "tests"):
        candidates += [p for p in (ROOT / folder).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".ipynb")]
    output = ROOT / "vihsd-source.zip"
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(output)


if __name__ == "__main__":
    main()
