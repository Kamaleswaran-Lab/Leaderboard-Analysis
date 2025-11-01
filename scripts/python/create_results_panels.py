#!/usr/bin/env python3
"""Orchestrate generation of all Results section figures."""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from create_results_figure1 import generate_figure1
from create_results_figure2 import generate_figure2
from create_results_figure3 import generate_figure3


def main() -> None:
    print("📊 CREATING RESULTS SECTION PANEL FIGURES")
    print("=" * 60)

    figure_paths = [
        generate_figure1(),
        generate_figure2(),
        generate_figure3(),
    ]

    print("\n" + "=" * 60)
    print("✅ ALL RESULTS FIGURES CREATED SUCCESSFULLY!\n")
    print("Generated files in figures/main_figures/:")
    for path in figure_paths:
        print(f"  • {path.name}")

    print("\nData sources used:")
    print("  ✓ Preliminary phase: optimal_threshold_results/")
    print("  ✓ Validation phase: optimal_threshold_results_validation/")
    print("  ✓ Factor weights: data/processed/factor_weights.csv")
    print("  ✓ Final leaderboard: data/raw/final_scores.csv")
    print("  ✗ No final-phase probability data used (per specification)")


if __name__ == "__main__":
    main()
