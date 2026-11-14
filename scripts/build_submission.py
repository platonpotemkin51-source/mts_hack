"""Package only the flat competition runner and its required dependency."""

import ast
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
        if name.endswith(".py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    if "httpx==0.28.1" not in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        raise SystemExit("Expected pinned httpx dependency is missing")
    output = ROOT / "runtime" / "submission.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            if name == "requirements.txt":
                archive.writestr(name, "httpx==0.28.1\n")
            else:
                archive.write(ROOT / name, arcname=name)
    if output.stat().st_size > 128 * 1024 * 1024:
        output.unlink()
        raise SystemExit("Submission ZIP exceeds the 128 MiB limit")
    with ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(FILES):
            raise SystemExit("Submission ZIP verification failed")
    print(f"Created verified flat submission ZIP: {output}")


if __name__ == "__main__":
    main()
