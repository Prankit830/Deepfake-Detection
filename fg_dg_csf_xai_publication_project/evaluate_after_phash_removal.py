import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ============================================================
# Project imports with compatibility fallback
# ============================================================

try:
    from config import cfg
    from data import DeepfakeDataModule, DeepfakeDataset, build_eval_transform
    from lightning_module import DetectorLightningModule as LitModel
    from lightning_module import compute_numpy_metrics
    from train import find_best_or_last_checkpoint, VARIANTS
except Exception:
    from config import cfg
    from data_module import DeepfakeDataModule, DeepfakeDataset, build_eval_transform
    from lit_model import FGDGCSFXAILightning as LitModel
    from lit_model import compute_numpy_metrics
    from train import find_best_or_last_checkpoint, VARIANTS


# ============================================================
# Helpers
# ============================================================

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_resolve(path):
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def make_loader(records, batch_size=None):
    if batch_size is None:
        batch_size = cfg.batch_size

    dataset = DeepfakeDataset(
        records,
        build_eval_transform(cfg.image_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader


def load_lit_model(run_name):
    ckpt_path = find_best_or_last_checkpoint(run_name)

    if ckpt_path is None:
        print(f"[SKIP] No checkpoint found for run: {run_name}")
        return None, None

    print(f"Loading {run_name} checkpoint:", ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lit_model = LitModel.load_from_checkpoint(
        str(ckpt_path),
        strict=True,
        map_location=device,
    )

    lit_model.to(device)
    lit_model.eval()

    return lit_model, ckpt_path


# ============================================================
# Load pHash-similar test image list
# ============================================================

def load_phash_similar_test_images(phash_csv_path):
    phash_csv_path = Path(phash_csv_path)

    if not phash_csv_path.exists():
        raise FileNotFoundError(
            f"pHash near-duplicate CSV not found: {phash_csv_path}\n"
            "Run this first:\n"
            "python forensic_audit.py --skip_face\n"
            "or\n"
            "python forensic_audit.py --phash_threshold 5 --face_similarity_threshold 0.65"
        )

    df = pd.read_csv(phash_csv_path)

    if "test_image" not in df.columns:
        raise ValueError(
            f"CSV must contain column 'test_image'. Found columns: {list(df.columns)}"
        )

    raw_paths = sorted(set(df["test_image"].dropna().astype(str).tolist()))
    resolved_paths = sorted(set(safe_resolve(p) for p in raw_paths))

    return df, raw_paths, resolved_paths


def filter_test_records(test_records, removed_raw_paths, removed_resolved_paths):
    removed_raw_set = set(removed_raw_paths)
    removed_resolved_set = set(removed_resolved_paths)

    clean_records = []
    removed_records = []

    for r in test_records:
        p_raw = str(r["path"])
        p_resolved = safe_resolve(p_raw)

        if p_raw in removed_raw_set or p_resolved in removed_resolved_set:
            removed_records.append(r)
        else:
            clean_records.append(r)

    return clean_records, removed_records


def records_to_df(records, split_name):
    rows = []

    for r in records:
        rows.append({
            "split": split_name,
            "path": r["path"],
            "label": int(r["label"]),
            "label_name": "fake" if int(r["label"]) == 1 else "real",
            "domain": int(r["domain"]),
            "domain_name": r.get("domain_name", ""),
        })

    return pd.DataFrame(rows)


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_loader(lit_model, loader):
    model = lit_model.model
    model.eval()

    device = next(model.parameters()).device

    labels = []
    probs = []
    paths = []

    for x, y, _domain, batch_paths in loader:
        x = x.to(device)

        logits = model(x)["logits"]

        prob_fake = F.softmax(
            logits,
            dim=1
        )[:, 1]

        labels.extend(y.numpy().tolist())
        probs.extend(prob_fake.detach().cpu().numpy().tolist())
        paths.extend(list(batch_paths))

    return {
        "labels": np.asarray(labels).astype(int),
        "probs_fake": np.asarray(probs).astype(float),
        "paths": paths,
    }


@torch.no_grad()
def predict_loader_tta(lit_model, loader, rounds=3):
    model = lit_model.model
    model.eval()

    device = next(model.parameters()).device

    labels = []
    probs = []
    paths = []

    for x, y, _domain, batch_paths in loader:
        x = x.to(device)

        probs_all = []

        for _ in range(rounds):
            logits = model(x)["logits"]

            probs_all.append(
                F.softmax(logits, dim=1)[:, 1]
            )

            x_flip = torch.flip(
                x,
                dims=[3]
            )

            logits_flip = model(x_flip)["logits"]

            probs_all.append(
                F.softmax(logits_flip, dim=1)[:, 1]
            )

        prob_mean = torch.stack(
            probs_all,
            dim=0
        ).mean(dim=0)

        labels.extend(y.numpy().tolist())
        probs.extend(prob_mean.detach().cpu().numpy().tolist())
        paths.extend(list(batch_paths))

    return {
        "labels": np.asarray(labels).astype(int),
        "probs_fake": np.asarray(probs).astype(float),
        "paths": paths,
    }


# ============================================================
# Plots
# ============================================================

def plot_confusion_matrix(y_true, y_prob, save_path, title):
    y_pred = (
        y_prob >= cfg.threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
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
                fontsize=13
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(y_true, y_prob, save_path, title):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_low_fpr_roc(y_true, y_prob, save_path, title):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.axvline(0.01, linestyle="--", label="1% FPR")
    plt.axvline(0.001, linestyle="--", label="0.1% FPR")
    plt.xlim(0, 0.05)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_score_distribution(y_true, y_prob, save_path, title):
    real_scores = y_prob[y_true == 0]
    fake_scores = y_prob[y_true == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(real_scores, bins=40, alpha=0.6, label="Real")
    plt.hist(fake_scores, bins=40, alpha=0.6, label="Fake")
    plt.xlabel("Predicted Fake Probability")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_failure_cases(preds, save_path, max_cases=12, image_size=160):
    labels = preds["labels"]
    probs = preds["probs_fake"]
    paths = preds["paths"]

    y_pred = (
        probs >= cfg.threshold
    ).astype(int)

    errors = np.where(
        y_pred != labels
    )[0]

    if len(errors) == 0:
        print("No failure cases found.")
        return

    conf = np.maximum(
        probs[errors],
        1.0 - probs[errors]
    )

    order = errors[
        np.argsort(-conf)
    ][:max_cases]

    cols = min(4, len(order))
    rows = int(math.ceil(len(order) / cols))

    plt.figure(figsize=(cols * 3.2, rows * 3.4))

    for i, idx in enumerate(order):
        img = Image.open(
            paths[idx]
        ).convert("RGB").resize(
            (image_size, image_size)
        )

        true_text = "Fake" if labels[idx] == 1 else "Real"
        pred_text = "Fake" if y_pred[idx] == 1 else "Real"

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(
            f"T:{true_text} P:{pred_text}\nFake p={probs[idx]:.2f}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle("Failure Cases After pHash-Similar Test Removal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# Evaluate one run
# ============================================================

def evaluate_run(run_name, loader, output_root, tta=False, tta_rounds=3):
    lit_model, ckpt_path = load_lit_model(run_name)

    if lit_model is None:
        return None

    run_dir = output_root / "runs" / run_name
    fig_dir = run_dir / "figures"
    table_dir = run_dir / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating cleaned test set: {run_name}")

    preds = predict_loader(
        lit_model,
        loader
    )

    metrics = compute_numpy_metrics(
        preds["labels"],
        preds["probs_fake"],
        cfg.threshold
    )

    metrics["run_name"] = run_name
    metrics["setting"] = "standard_cleaned_test"
    metrics["checkpoint"] = str(ckpt_path)

    save_json(
        metrics,
        table_dir / "cleaned_test_metrics.json"
    )

    pd.DataFrame([metrics]).to_csv(
        table_dir / "cleaned_test_metrics.csv",
        index=False
    )

    pd.DataFrame({
        "path": preds["paths"],
        "label": preds["labels"],
        "prob_fake": preds["probs_fake"],
    }).to_csv(
        table_dir / "cleaned_test_predictions.csv",
        index=False
    )

    plot_confusion_matrix(
        preds["labels"],
        preds["probs_fake"],
        fig_dir / "cleaned_confusion_matrix.png",
        f"{run_name} - Cleaned Confusion Matrix"
    )

    plot_roc(
        preds["labels"],
        preds["probs_fake"],
        fig_dir / "cleaned_roc_curve.png",
        f"{run_name} - Cleaned ROC"
    )

    plot_low_fpr_roc(
        preds["labels"],
        preds["probs_fake"],
        fig_dir / "cleaned_low_fpr_roc.png",
        f"{run_name} - Cleaned Low-FPR ROC"
    )

    plot_score_distribution(
        preds["labels"],
        preds["probs_fake"],
        fig_dir / "cleaned_score_distribution.png",
        f"{run_name} - Cleaned Score Distribution"
    )

    save_failure_cases(
        preds,
        fig_dir / "cleaned_failure_cases.png"
    )

    rows = [metrics]

    if tta:
        print(f"Evaluating cleaned test set with TTA: {run_name}")

        tta_preds = predict_loader_tta(
            lit_model,
            loader,
            rounds=tta_rounds
        )

        tta_metrics = compute_numpy_metrics(
            tta_preds["labels"],
            tta_preds["probs_fake"],
            cfg.threshold
        )

        tta_metrics["run_name"] = run_name
        tta_metrics["setting"] = "tta_cleaned_test"
        tta_metrics["checkpoint"] = str(ckpt_path)

        save_json(
            tta_metrics,
            table_dir / "cleaned_test_tta_metrics.json"
        )

        pd.DataFrame([tta_metrics]).to_csv(
            table_dir / "cleaned_test_tta_metrics.csv",
            index=False
        )

        pd.DataFrame({
            "path": tta_preds["paths"],
            "label": tta_preds["labels"],
            "prob_fake": tta_preds["probs_fake"],
        }).to_csv(
            table_dir / "cleaned_test_tta_predictions.csv",
            index=False
        )

        rows.append(tta_metrics)

    return rows


# ============================================================
# Compare old vs cleaned metrics
# ============================================================

def load_old_metrics_for_run(run_name):
    path = Path(cfg.output_dir) / "runs" / run_name / "test_metrics.json"

    if not path.exists():
        return None

    with open(path) as f:
        m = json.load(f)

    m["run_name"] = run_name
    m["setting"] = "original_test"

    return m


def create_old_vs_cleaned_comparison(summary_df, output_root):
    rows = []

    for run_name in summary_df["run_name"].unique():
        old = load_old_metrics_for_run(run_name)

        if old:
            rows.append(old)

        clean_rows = summary_df[
            summary_df["run_name"] == run_name
        ].to_dict("records")

        rows.extend(clean_rows)

    if not rows:
        return

    comp = pd.DataFrame(rows)

    comp.to_csv(
        output_root / "tables" / "original_vs_cleaned_metrics_comparison.csv",
        index=False
    )


# ============================================================
# ZIP
# ============================================================

def create_zip(output_root):
    zip_path = Path("phash_cleaned_evaluation_outputs.zip")

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        str(zip_path).replace(".zip", ""),
        "zip",
        root_dir=str(output_root)
    )

    print("Created ZIP:", zip_path)

    return zip_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phash_csv",
        type=str,
        default="outputs/forensic_leakage_audit/tables/train_test_phash_near_duplicates.csv",
        help="CSV produced by forensic_audit.py containing test_image column."
    )

    parser.add_argument(
        "--expected_removed",
        type=int,
        default=355,
        help="Expected number of unique pHash-similar test images."
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=["final"],
        help="Runs to evaluate. Use 'all' to evaluate every trained run."
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Also run TTA on cleaned test set."
    )

    parser.add_argument(
        "--tta_rounds",
        type=int,
        default=3
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/phash_cleaned_evaluation"
    )

    args = parser.parse_args()

    output_root = Path(args.output_dir)
    table_root = output_root / "tables"

    table_root.mkdir(parents=True, exist_ok=True)

    # Load original DataModule and test records
    dm = DeepfakeDataModule()
    dm.setup()

    original_test_records = dm.test_records

    phash_df, removed_raw, removed_resolved = load_phash_similar_test_images(
        args.phash_csv
    )

    clean_records, removed_records = filter_test_records(
        original_test_records,
        removed_raw,
        removed_resolved
    )

    print("\nOriginal unseen test images:", len(original_test_records))
    print("Unique pHash-similar test images from CSV:", len(removed_resolved))
    print("Actually removed from current test split:", len(removed_records))
    print("Cleaned unseen test images:", len(clean_records))

    if len(removed_resolved) != args.expected_removed:
        print(
            f"WARNING: expected {args.expected_removed} unique pHash-similar test images, "
            f"but CSV contains {len(removed_resolved)} unique test images."
        )

    if len(removed_records) != args.expected_removed:
        print(
            f"WARNING: expected to remove {args.expected_removed} images from current split, "
            f"but removed {len(removed_records)}. "
            "This can happen if the pHash CSV was generated using a different split."
        )

    # Save removed and cleaned split files
    records_to_df(
        original_test_records,
        "original_unseen_test"
    ).to_csv(
        table_root / "original_unseen_test_split.csv",
        index=False
    )

    records_to_df(
        removed_records,
        "removed_phash_similar_test"
    ).to_csv(
        table_root / "removed_phash_similar_test_images.csv",
        index=False
    )

    records_to_df(
        clean_records,
        "cleaned_unseen_test"
    ).to_csv(
        table_root / "cleaned_unseen_test_split.csv",
        index=False
    )

    phash_df.to_csv(
        table_root / "source_phash_near_duplicate_pairs.csv",
        index=False
    )

    removal_summary = {
        "original_test_images": len(original_test_records),
        "unique_phash_similar_test_images_in_csv": len(removed_resolved),
        "actually_removed_from_current_split": len(removed_records),
        "cleaned_test_images": len(clean_records),
        "expected_removed": args.expected_removed,
        "phash_csv": str(args.phash_csv),
    }

    save_json(
        removal_summary,
        table_root / "phash_cleaning_summary.json"
    )

    pd.DataFrame([removal_summary]).to_csv(
        table_root / "phash_cleaning_summary.csv",
        index=False
    )

    # Prepare loader
    clean_loader = make_loader(
        clean_records
    )

    # Runs
    if len(args.runs) == 1 and args.runs[0].lower() == "all":
        run_names = list(VARIANTS.keys())
    else:
        run_names = args.runs

    all_rows = []

    for run_name in run_names:
        result_rows = evaluate_run(
            run_name,
            clean_loader,
            output_root,
            tta=args.tta,
            tta_rounds=args.tta_rounds
        )

        if result_rows:
            all_rows.extend(result_rows)

    if len(all_rows) == 0:
        raise RuntimeError("No model runs were evaluated. Check checkpoints.")

    summary_df = pd.DataFrame(all_rows)

    summary_df.to_csv(
        table_root / "cleaned_test_all_runs_summary.csv",
        index=False
    )

    create_old_vs_cleaned_comparison(
        summary_df,
        output_root
    )

    zip_path = create_zip(
        output_root
    )

    print("\nDone.")
    print("Cleaned evaluation output folder:", output_root)
    print("ZIP:", zip_path)


if __name__ == "__main__":
    main()