#!/usr/bin/env python3
"""Generate Figures 1–3 in separate processes to preserve each figure's style."""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    directory = Path(__file__).resolve().parent
    for filename in ["create_results_figure1.py", "create_results_figure2.py", "create_results_figure3.py"]:
        print(f"Generating {filename}", flush=True)
        subprocess.run([sys.executable, str(directory / filename)], check=True)
    print("Saved Figures 1–3 in figures/main_figures/.")


if __name__ == "__main__":
    main()
