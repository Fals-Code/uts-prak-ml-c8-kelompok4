"""Siapkan gambar cuplikan berkas/log aktual, bukan screenshot antarmuka."""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

import preprocessing

BASE = Path(__file__).resolve().parent


def text_image(text, title, path, font_size=10):
    lines = text.splitlines()
    figure = plt.figure(figsize=(12, max(3, len(lines) * 0.20 + 1)))
    figure.text(0.025, 0.97, title, va="top", fontsize=14, weight="bold")
    figure.text(0.025, 0.91, text, va="top", family="monospace", fontsize=font_size)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white",
                   metadata={"Software": "Week-09 literal code/output evidence"})
    plt.close(figure)


def main():
    evidence = BASE / "outputs/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    commands = [
        [sys.executable, str(BASE / "preprocessing.py")],
        [sys.executable, "-m", "unittest", "discover", "-s", str(BASE),
         "-p", "test_preprocessing.py", "-v"],
    ]
    logs = []
    for command in commands:
        completed = subprocess.run(command, cwd=BASE, env=environment,
                                   capture_output=True, encoding="utf-8")
        logs.append(f"COMMAND: {' '.join(command)}\nEXIT CODE: {completed.returncode}\n"
                    + completed.stdout + completed.stderr)
        if completed.returncode:
            (evidence / "validation.txt").write_text("\n".join(logs), encoding="utf-8")
            raise SystemExit("Pembuatan bukti dihentikan: program/tes gagal, lihat validation.txt.")
    (evidence / "validation.txt").write_text("\n".join(logs), encoding="utf-8")
    code = inspect.getsource(preprocessing.prepare_mall)
    source_lines = (BASE / "preprocessing.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source_lines, 1) if line.startswith("def prepare_mall"))
    numbered = "\n".join(f"{number:3}  {line}" for number, line in enumerate(code.splitlines(), start))
    text_image(numbered, "Cuplikan literal preprocessing.py: fitur Mall dan referensi baris",
               evidence / "code_preprocessing.png", font_size=9)
    short_log = logs[0].split("\n", 1)[1]
    mall = pd.read_csv(BASE / "data/processed/mall_kmodes.csv", dtype=str)
    telco = pd.read_csv(BASE / "data/processed/telco_kmodes.csv", dtype=str)
    output = short_log + "\nHEAD MALL (CSV dibaca ulang):\n" + mall.head().to_string(index=False)
    output += "\n\nHEAD TELCO (4 dari 12 fitur, CSV dibaca ulang):\n"
    output += telco[["Contract", "InternetService", "TenureCategory", "MonthlyChargesCategory"]].head().to_string(index=False)
    text_image(output, "Output eksekusi aktual dan lima baris fitur kategori", evidence / "output_preprocessing.png")
    print("Bukti kode/output serta log empat tes tersimpan di outputs/evidence.")


if __name__ == "__main__":
    main()
