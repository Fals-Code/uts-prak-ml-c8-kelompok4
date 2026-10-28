"""Jalankan program dua kali ke folder keluaran berbeda, lalu bandingkan hash CSV/JSON."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = [BASE / "outputs_run1", BASE / "outputs_run2"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for out in RUNS:
        env = dict(os.environ, W9_OUT_DIR=str(out))
        subprocess.run([sys.executable, str(BASE / "program_clustering.py"),
                        "--stage", "all", "--dataset", "all"], check=True, env=env)

    files = sorted(p.relative_to(RUNS[0] / "tables") for p in (RUNS[0] / "tables").glob("*")
                   if p.suffix in {".csv", ".json"})
    lines, ok = [], True
    for rel in files:
        a, b = RUNS[0] / "tables" / rel, RUNS[1] / "tables" / rel
        same = b.exists() and sha(a) == sha(b)
        ok &= same
        lines.append(f"{'SAMA ' if same else 'BEDA '} {rel}")
    lines.append(f"\nHASIL: {'SEMUA IDENTIK' if ok else 'ADA PERBEDAAN'}")
    report = "\n".join(lines)
    print(report)
    ev = BASE / "outputs" / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "reproducibility.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
