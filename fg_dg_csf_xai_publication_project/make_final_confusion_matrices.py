import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from config import cfg


RUN_NAME = "final"
OUTPUT_ROOT = Path("outputs/final_confusion_matrices")
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
ZIP_NAME = "final_confusion_matrices_with_without_tta.zip"


def make_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(pred_path):
    pred_path = Path(pred_path)

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n\n"
            "Run this first:\n"
            "python train.py --mode final --final_epochs 12 --tta --leakage_check"
        )

    df = pd.read_csv(pred_path)

    required_cols = {"label", "prob_fake"}

    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"{pred_path} must contain columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )

    y_true = df["label"].values.astype(int)
    y_prob = df["prob_fake"].values.astype(float)
    y_pred = (y_prob >= cfg.threshold).astype(int)

    return df, y_true, y_prob, y_pred


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_matrix_table(cm, setting_name):
    table = pd.DataFrame(
        cm,
        index=["Actual Real", "Actual Fake"],
        columns=["Predicted Real", "Predicted Fake"]
    )

    table.to_csv(
        TABLE_DIR / f"confusion_matrix_{setting_name}.csv"
    )

    return table


def plot_confusion_matrix(cm, setting_name, title):
    plt.figure(figsize=(6, 5))

    plt.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold"
            )

    plt.colorbar()
    plt.tight_layout()

    save_path = FIGURE_DIR / f"confusion_matrix_{setting_name}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_side_by_side(cm_standard, cm_tta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [
        (cm_standard, "Without TTA"),
        (cm_tta, "With TTA")
    ]

    for ax, (cm, title) in zip(axes, matrices):
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Real", "Fake"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold"
                )

    fig.suptitle("Full Proposed Model Confusion Matrices: Without TTA vs With TTA")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    save_path = FIGURE_DIR / "confusion_matrix_standard_vs_tta.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def create_zip():
    zip_path = Path(ZIP_NAME)

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        str(zip_path).replace(".zip", ""),
        "zip",
        root_dir=str(OUTPUT_ROOT)
    )

    return zip_path


def main():
    make_dirs()

    standard_pred_path = Path("outputs") / "runs" / RUN_NAME / "test_predictions.csv"
    tta_pred_path = Path("outputs") / "runs" / RUN_NAME / "tta_predictions.csv"

    print("Loading standard predictions:")
    print(standard_pred_path)

    standard_df, y_true_std, y_prob_std, y_pred_std = load_predictions(
        standard_pred_path
    )

    print("Loading TTA predictions:")
    print(tta_pred_path)

    tta_df, y_true_tta, y_prob_tta, y_pred_tta = load_predictions(
        tta_pred_path
    )

    cm_standard = confusion_matrix(
        y_true_std,
        y_pred_std,
        labels=[0, 1]
    )

    cm_tta = confusion_matrix(
        y_true_tta,
        y_pred_tta,
        labels=[0, 1]
    )

    standard_metrics = calculate_metrics(
        y_true_std,
        y_pred_std
    )

    tta_metrics = calculate_metrics(
        y_true_tta,
        y_pred_tta
    )

    standard_metrics["Setting"] = "Without TTA"
    tta_metrics["Setting"] = "With TTA"

    comparison_df = pd.DataFrame([
        standard_metrics,
        tta_metrics
    ])

    comparison_df = comparison_df[
        [
            "Setting",
            "TN",
            "FP",
            "FN",
            "TP",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
    ]

    comparison_df.to_csv(
        TABLE_DIR / "confusion_matrix_metrics_standard_vs_tta.csv",
        index=False
    )

    save_confusion_matrix_table(
        cm_standard,
        "without_tta"
    )

    save_confusion_matrix_table(
        cm_tta,
        "with_tta"
    )

    plot_confusion_matrix(
        cm_standard,
        "without_tta",
        "Full Proposed Model Confusion Matrix Without TTA"
    )

    plot_confusion_matrix(
        cm_tta,
        "with_tta",
        "Full Proposed Model Confusion Matrix With TTA"
    )

    plot_side_by_side(
        cm_standard,
        cm_tta
    )

    summary = {
        "run_name": RUN_NAME,
        "threshold": cfg.threshold,
        "standard_prediction_file": str(standard_pred_path),
        "tta_prediction_file": str(tta_pred_path),
        "standard_confusion_matrix": cm_standard.tolist(),
        "tta_confusion_matrix": cm_tta.tolist(),
        "standard_metrics": standard_metrics,
        "tta_metrics": tta_metrics,
    }

    with open(TABLE_DIR / "confusion_matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    zip_path = create_zip()

    print("\nDone.")
    print("Saved tables in:", TABLE_DIR)
    print("Saved figures in:", FIGURE_DIR)
    print("ZIP file:", zip_path)

    print("\nComparison:")
    print(comparison_df)


if __name__ == "__main__":
    main()