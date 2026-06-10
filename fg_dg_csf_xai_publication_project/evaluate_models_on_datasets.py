import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from config import cfg
from data import scan_dataset_root, ensure_splits, DeepfakeDataset, build_eval_transform
from lightning_module import DetectorLightningModule, compute_numpy_metrics


DEFAULT_RUNS = [
    "final",
    "freqnet_like_baseline",
    "f3net_like_baseline",
    "mesonet_baseline",
    "efficientnet_b4_baseline",
    "convnext_tiny_baseline",
    "vit_b16_baseline",
    "swin_tiny_baseline",
    "xception_baseline",
    "spatial_baseline",
    "spatial_frequency",
    "spatial_frequency_patch",
    "no_counterfactual",
    "no_domain",
]


def parse_dataset_args(dataset_args):
    datasets = []
    for item in dataset_args:
        if "=" not in item:
            raise ValueError(
                f"Dataset argument must be Name=/path format, got: {item}"
            )
        name, root = item.split("=", 1)
        datasets.append((name.strip(), Path(root).expanduser().resolve()))
    return datasets


def find_checkpoint(run_name):
    ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    if not ckpt_dir.exists():
        return None

    best = sorted(ckpt_dir.glob("best-*.ckpt"))
    if best:
        return best[-1]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    return None


def make_loader(records):
    dataset = DeepfakeDataset(records, build_eval_transform(cfg.image_size))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_dataset_records(dataset_name, dataset_root, use_all=False, max_samples=None):
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    records = scan_dataset_root(dataset_root, domain_id=99, domain_name=dataset_name)
    if len(records) == 0:
        raise RuntimeError(
            f"No labeled images found in {dataset_root}. Folder names must contain real/fake aliases. "
            "For FaceForensics++/Celeb-DF, use extracted frame folders with real/original and fake/manipulated names."
        )

    if use_all:
        selected = records
    else:
        splits = ensure_splits(records, seed=cfg.seed + 500)
        selected = splits["test"] if len(splits["test"]) else records

    if max_samples is not None and len(selected) > max_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(selected), size=max_samples, replace=False)
        selected = [selected[i] for i in idx]

    return selected


def load_model(run_name, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint(run_name)

    if checkpoint_path is None:
        return None, None

    model = DetectorLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        strict=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict_loader(lit_model, loader, device):
    probs = []
    labels = []
    paths = []

    for x, y, _domain, batch_paths in loader:
        x = x.to(device)
        logits = lit_model.model(x)["logits"]
        p = F.softmax(logits, dim=1)[:, 1]

        probs.extend(p.detach().cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
        paths.extend(list(batch_paths))

    return np.asarray(labels).astype(int), np.asarray(probs).astype(float), paths


def plot_roc(y_true, y_prob, save_path, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion(y_true, y_prob, save_path, title):
    y_pred = (y_prob >= cfg.threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_score_distribution(y_true, y_prob, save_path, title):
    plt.figure(figsize=(7, 5))
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Real")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake")
    plt.xlabel("Predicted Fake Probability")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_auc_bar(summary_df, output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in summary_df["dataset"].unique():
        sub = summary_df[summary_df["dataset"] == dataset_name].copy()
        if "auc" not in sub.columns:
            continue
        sub = sub.sort_values("auc", ascending=False)
        plt.figure(figsize=(11, 5))
        plt.bar(sub["run_name"], sub["auc"])
        plt.ylabel("AUC")
        plt.title(f"Model Comparison on {dataset_name}")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        safe_dataset = dataset_name.replace(" ", "_").replace("+", "plus")
        plt.savefig(fig_dir / f"model_auc_bar_{safe_dataset}.png", dpi=300)
        plt.close()


def plot_auc_heatmap(summary_df, output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pivot = summary_df.pivot_table(index="dataset", columns="run_name", values="auc", aggfunc="mean")
    if pivot.empty:
        return

    plt.figure(figsize=(max(10, 0.7 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="AUC")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("AUC Heatmap Across Models and Datasets")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "model_dataset_auc_heatmap.png", dpi=300)
    plt.close()


def evaluate_one(run_name, dataset_name, records, output_dir, checkpoint_path=None):
    lit_model, device = load_model(run_name, checkpoint_path=checkpoint_path)
    if lit_model is None:
        print(f"Skipping {run_name}; checkpoint not found.")
        return None

    loader = make_loader(records)
    y_true, y_prob, paths = predict_loader(lit_model, loader, device)
    metrics = compute_numpy_metrics(y_true, y_prob, cfg.threshold)

    safe_dataset = dataset_name.replace(" ", "_").replace("+", "plus")
    run_dir = Path(output_dir) / "runs" / run_name / safe_dataset
    fig_dir = run_dir / "figures"
    table_dir = run_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"path": paths, "label": y_true, "prob_fake": y_prob}).to_csv(
        table_dir / "predictions.csv", index=False
    )
    with open(table_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([{**metrics, "dataset": dataset_name, "run_name": run_name}]).to_csv(
        table_dir / "metrics.csv", index=False
    )

    plot_roc(y_true, y_prob, fig_dir / "roc_curve.png", f"{run_name} on {dataset_name}")
    plot_confusion(y_true, y_prob, fig_dir / "confusion_matrix.png", f"{run_name} on {dataset_name}")
    plot_score_distribution(y_true, y_prob, fig_dir / "score_distribution.png", f"{run_name} on {dataset_name}")

    return {
        "dataset": dataset_name,
        "run_name": run_name,
        "n_images": int(len(y_true)),
        **metrics,
    }


def create_zip(output_dir):
    output_dir = Path(output_dir)
    zip_path = Path("external_model_dataset_evaluation.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", root_dir=str(output_dir))
    print("Created ZIP:", zip_path)
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset list in Name=/path format. Example: FaceForensics++=/data/ffpp Celeb-DF=/data/celebdf",
    )
    parser.add_argument("--runs", nargs="+", default=["all"], help="Run names to evaluate, or 'all'.")
    parser.add_argument("--use_all", action="store_true", help="Use all images instead of auto test split.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per dataset.")
    parser.add_argument("--output_dir", default="external_model_dataset_evaluation_outputs")
    args = parser.parse_args()

    datasets = parse_dataset_args(args.datasets)
    runs = DEFAULT_RUNS if args.runs == ["all"] else args.runs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json = lambda obj, path: (Path(path).parent.mkdir(parents=True, exist_ok=True), Path(path).write_text(json.dumps(obj, indent=2)))
    save_json({"datasets": [(n, str(p)) for n, p in datasets], "runs": runs}, output_dir / "evaluation_config.json")

    rows = []
    dataset_info_rows = []

    for dataset_name, dataset_root in datasets:
        print("Preparing dataset:", dataset_name, dataset_root)
        records = prepare_dataset_records(
            dataset_name,
            dataset_root,
            use_all=args.use_all,
            max_samples=args.max_samples,
        )
        labels = np.asarray([r["label"] for r in records])
        dataset_info_rows.append({
            "dataset": dataset_name,
            "root": str(dataset_root),
            "n_images": len(records),
            "real": int(np.sum(labels == 0)),
            "fake": int(np.sum(labels == 1)),
        })

        for run_name in runs:
            print(f"Evaluating {run_name} on {dataset_name}")
            result = evaluate_one(run_name, dataset_name, records, output_dir)
            if result is not None:
                rows.append(result)

    pd.DataFrame(dataset_info_rows).to_csv(output_dir / "dataset_summary.csv", index=False)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "model_dataset_performance_summary.csv", index=False)

    if len(summary_df):
        plot_auc_bar(summary_df, output_dir)
        plot_auc_heatmap(summary_df, output_dir)

    create_zip(output_dir)
    print("Done. Summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
