"""Flatten the application package for the organizer's submission format."""

import ast
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = {name: f"energytool/{name}" for name in (
    "solution.py", "contracts.py", "consultant.py", "llm_client.py",
    "kb_loader.py", "chunking.py", "retrieval.py",
)}
FILES["requirements.txt"] = "requirements.txt"


def submission_file(name: str) -> bytes:
    if name == "requirements.txt":
        return b"httpx==0.28.1\n"
    source = (ROOT / FILES[name]).read_text(encoding="utf-8")
    # The platform allows only flat modules; repository imports stay packaged.
    source = re.sub(r"(?m)^from energytool\.(\w+) import ", r"from \1 import ", source)
    ast.parse(source, filename=name)
    return source.encode("utf-8")


def main() -> None:
    for path in FILES.values():
        if not (ROOT / path).is_file():
            raise SystemExit(f"Missing required source file: {path}")
    if "httpx==0.28.1" not in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        raise SystemExit("Expected pinned httpx dependency is missing")
    output = ROOT / "runtime" / "submission.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.writestr(name, submission_file(name))
    if output.stat().st_size > 128 * 1024 * 1024:
        output.unlink()
        raise SystemExit("Submission ZIP exceeds the 128 MiB limit")
    with ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(FILES):
            raise SystemExit("Submission ZIP verification failed")
    print(f"Created verified flat submission ZIP: {output}")


if __name__ == "__main__":
    main()
