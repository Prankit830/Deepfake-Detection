"""
Use this file only if you have official F3-Net, FreqNet, SBI, RECCE prediction CSV files.

Expected CSV format:
path,label,prob_fake

Place files here:
outputs/official_baseline_predictions/F3Net.csv
outputs/official_baseline_predictions/FreqNet.csv
outputs/official_baseline_predictions/SBI.csv
outputs/official_baseline_predictions/RECCE.csv

Then run:
python official_baseline_import.py
"""

from pathlib import Path

import pandas as pd

from config import cfg
from lightning_module import compute_numpy_metrics


def main():
    input_dir = Path(cfg.output_dir) / "official_baseline_predictions"
    table_dir = Path(cfg.output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if not input_dir.exists():
        print("Folder not found:", input_dir)
        print("Create it and add CSV files with path,label,prob_fake.")
        return

    for csv_file in input_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)

        required = {"label", "prob_fake"}

        if not required.issubset(set(df.columns)):
            print("Skipping", csv_file, "missing label/prob_fake")
            continue

        metrics = compute_numpy_metrics(
            df["label"].values,
            df["prob_fake"].values,
            cfg.threshold,
        )

        rows.append({
            "Model": csv_file.stem,
            "Variant": f"official_{csv_file.stem}",
            **metrics,
        })

    if not rows:
        print("No official baseline CSVs processed.")
        return

    official_df = pd.DataFrame(rows)

    official_df.to_csv(
        table_dir / "official_baseline_comparison.csv",
        index=False,
    )

    baseline_path = table_dir / "baseline_comparison.csv"

    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        combined = pd.concat([existing, official_df], ignore_index=True)
    else:
        combined = official_df

    combined.to_csv(
        table_dir / "baseline_comparison_with_official_methods.csv",
        index=False,
    )

    print("Saved official baseline tables.")


if __name__ == "__main__":
    main()
