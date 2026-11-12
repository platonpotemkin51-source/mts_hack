"""Package the flat source scaffold, NOT a ready competition solution."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "solution.py",
    "requirements.txt",
    "contracts.py",
    "consultant.py",
    "llm_client.py",
    "kb_loader.py",
    "chunking.py",
    "retrieval.py",
)


def main() -> None:
    for name in FILES:
        if not (ROOT / name).is_file():
            raise SystemExit(f"Missing required source file: {name}")
    output = ROOT / "runtime" / "submission-scaffold.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
    print(f"Created source scaffold: {output}")
    print("NOT READY FOR EVALUATION: solution.py has not been implemented.")


if __name__ == "__main__":
    main()
