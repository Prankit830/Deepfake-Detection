import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from config import cfg
from data import scan_dataset_root, ensure_splits, DeepfakeDataset, build_eval_transform
from lightning_module import compute_numpy_metrics
from train import load_lightning_model


def make_loader(records):
    return DataLoader(
        DeepfakeDataset(records, build_eval_transform(cfg.image_size)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(lit_model, loader):
    device = next(lit_model.model.parameters()).device
    model = lit_model.model
    model.eval()

    probs = []
    labels = []
    paths = []

    for x, y, _domain, batch_paths in loader:
        x = x.to(device)
        logits = model(x)["logits"]
        p = F.softmax(logits, dim=1)[:, 1]

        probs.extend(p.detach().cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
        paths.extend(list(batch_paths))

    return np.asarray(labels).astype(int), np.asarray(probs).astype(float), paths


def plot_roc(y_true, y_prob, save_path, title):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_dataset(dataset_name, dataset_root, run_name):
    root = Path(dataset_root)

    records = scan_dataset_root(root, domain_id=99, domain_name=dataset_name)

    if len(records) == 0:
        raise RuntimeError("No labeled images found. Folder names must contain real/fake aliases.")

    splits = ensure_splits(records, seed=cfg.seed + 500)

    test_records = splits["test"] if len(splits["test"]) else records

    loader = make_loader(test_records)

    lit_model = load_lightning_model(run_name)

    if lit_model is None:
        raise FileNotFoundError(f"No checkpoint found for run: {run_name}")

    y_true, y_prob, paths = predict(lit_model, loader)

    metrics = compute_numpy_metrics(y_true, y_prob, cfg.threshold)

    out_dir = Path(cfg.output_dir) / "external_dataset_eval"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    safe_name = dataset_name.replace(" ", "_").replace("+", "plus")

    with open(table_dir / f"{safe_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([{"dataset": dataset_name, **metrics}]).to_csv(
        table_dir / f"{safe_name}_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "path": paths,
        "label": y_true,
        "prob_fake": y_prob,
    }).to_csv(
        table_dir / f"{safe_name}_predictions.csv",
        index=False,
    )

    plot_roc(
        y_true,
        y_prob,
        fig_dir / f"{safe_name}_roc_curve.png",
        f"ROC Curve on {dataset_name}",
    )

    print(dataset_name, "metrics:", metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--run_name", default="final")
    args = parser.parse_args()

    evaluate_dataset(args.dataset_name, args.dataset_root, args.run_name)


if __name__ == "__main__":
    main()
