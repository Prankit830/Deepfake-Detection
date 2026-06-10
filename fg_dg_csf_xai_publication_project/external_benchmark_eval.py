import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

from config import cfg
from data import (
    scan_dataset_root,
    DeepfakeDataset,
    build_eval_transform,
)

from torch.utils.data import DataLoader

from lightning_module import DetectorLightningModule


# ============================================================
# CONFIG
# ============================================================

BENCHMARKS = {
    "FaceForensics++": cfg.EXTERNAL_BENCHMARKS["FaceForensics++"],
    "Celeb-DF": cfg.EXTERNAL_BENCHMARKS["Celeb-DF"],
    "DFDC": cfg.EXTERNAL_BENCHMARKS["DFDC"],
}

OUTPUT_DIR = Path("external_benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint_path = cfg.best_checkpoint

model = DetectorLightningModule.load_from_checkpoint(
    checkpoint_path,
    map_location=DEVICE
)

model.eval()
model.to(DEVICE)

print("Loaded model:", checkpoint_path)


# ============================================================
# CREATE DATALOADER
# ============================================================

def make_loader(records):

    dataset = DeepfakeDataset(
        records,
        transform=build_eval_transform(cfg.image_size)
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    y_true = []
    y_prob = []

    for images, labels, domains, paths in loader:

        images = images.to(DEVICE)

        outputs = model.model(images)

        logits = outputs["logits"]

        probs = F.softmax(logits, dim=1)[:, 1]

        y_true.extend(labels.numpy())
        y_prob.extend(
            probs.cpu().numpy()
        )

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":
            accuracy_score(y_true, y_pred),

        "auc":
            roc_auc_score(y_true, y_prob),

        "f1":
            f1_score(y_true, y_pred),

        "precision":
            precision_score(y_true, y_pred),

        "recall":
            recall_score(y_true, y_pred),
    }

    return metrics, y_true, y_prob


# ============================================================
# PLOTS
# ============================================================

def plot_roc(y_true, y_prob, save_path):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    auc_score = roc_auc_score(
        y_true,
        y_prob
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc_score:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_prob, save_path):

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(cm)

    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.title("Confusion Matrix")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

all_results = []

for benchmark_name, benchmark_path in BENCHMARKS.items():

    print("\n" + "=" * 60)
    print("Evaluating:", benchmark_name)
    print("=" * 60)

    records = scan_dataset_root(
        benchmark_path,
        domain_id=999,
        domain_name=benchmark_name
    )

    print("Images found:", len(records))

    loader = make_loader(records)

    metrics, y_true, y_prob = evaluate(loader)

    print(metrics)

    benchmark_dir = OUTPUT_DIR / benchmark_name
    benchmark_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        benchmark_dir / "metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    # --------------------------------------------------------
    # Save ROC
    # --------------------------------------------------------

    plot_roc(
        y_true,
        y_prob,
        benchmark_dir / "roc_curve.png"
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    plot_confusion(
        y_true,
        y_prob,
        benchmark_dir / "confusion_matrix.png"
    )

    # --------------------------------------------------------
    # Add summary row
    # --------------------------------------------------------

    row = {
        "Dataset": benchmark_name,
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
    }

    all_results.append(row)

# ============================================================
# FINAL TABLE
# ============================================================

df = pd.DataFrame(all_results)

df.to_csv(
    OUTPUT_DIR / "external_benchmark_summary.csv",
    index=False
)

print("\nSaved summary table.")

print(df)