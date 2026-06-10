import json
import math
import shutil
import hashlib
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix

import torch
import torch.nn.functional as F

from config import cfg
from data import build_eval_transform
from lightning_module import compute_numpy_metrics


def save_training_config_table(table_dir):
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = [{"Parameter": k, "Value": str(v)} for k, v in cfg.__dict__.items()]

    pd.DataFrame(rows).to_csv(
        table_dir / "training_configuration.csv",
        index=False,
    )


def create_architecture_diagram(fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    boxes = [
        ("Input\n224×224 RGB", (0.06, 0.55)),
        ("Spatial Branch\nResNet50\nvisible artifacts", (0.25, 0.75)),
        ("Frequency Branch\nFFT + ResNet18\nspectral artifacts", (0.25, 0.55)),
        ("Patch-Frequency\nlocal spectral cues", (0.25, 0.35)),
        ("Gated Fusion\n512-D fused feature", (0.53, 0.55)),
        ("Classifier\nReal / Fake", (0.76, 0.66)),
        ("Domain Head\nGRL during training", (0.76, 0.42)),
        ("Explainability\nGrad-CAM + FFT", (0.90, 0.55)),
    ]

    for text, (x, y) in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black"),
        )

    arrows = [
        ((0.11, 0.55), (0.20, 0.75)),
        ((0.11, 0.55), (0.20, 0.55)),
        ((0.11, 0.55), (0.20, 0.35)),
        ((0.34, 0.75), (0.45, 0.57)),
        ((0.34, 0.55), (0.45, 0.55)),
        ((0.34, 0.35), (0.45, 0.53)),
        ((0.61, 0.55), (0.69, 0.66)),
        ((0.61, 0.55), (0.69, 0.42)),
        ((0.82, 0.66), (0.86, 0.55)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    ax.set_title(
        "Proposed FG-DG-CSF-XAI Architecture",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir / "proposed_architecture_diagram.png",
        dpi=300,
    )

    plt.close()


def read_metrics_csv(run_name):
    base = Path(cfg.output_dir) / "runs" / run_name / "logs"
    candidates = list(base.rglob("metrics.csv"))

    if not candidates:
        return None

    return pd.read_csv(candidates[0])


def plot_training_curves(run_name, fig_dir):
    df = read_metrics_csv(run_name)

    if df is None:
        print("No metrics.csv found for training curves.")
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def pick(col_names):
        for c in col_names:
            if c in df.columns:
                return c
        return None

    x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

    curve_specs = [
        ("Training Curves: Loss", ["train_loss", "train_loss_epoch"], ["val_loss"], "training_loss_curve.png", "Loss"),
        ("Training Curves: Accuracy", ["train_acc", "train_acc_epoch"], ["val_acc"], "training_accuracy_curve.png", "Accuracy"),
        ("Training Curves: AUC", ["train_auc"], ["val_auc"], "training_auc_curve.png", "AUC"),
    ]

    for title, train_cols, val_cols, fname, ylabel in curve_specs:
        train_col = pick(train_cols)
        val_col = pick(val_cols)

        plt.figure(figsize=(7, 5))

        if train_col:
            plt.plot(x, df[train_col], marker="o", label=train_col)

        if val_col:
            plt.plot(x, df[val_col], marker="o", label=val_col)

        plt.xlabel("Epoch / Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=300)
        plt.close()

    hardness_col = pick(["train_hardness", "train_hardness_epoch"])

    if hardness_col:
        plt.figure(figsize=(7, 5))
        plt.plot(x, df[hardness_col], marker="o")
        plt.xlabel("Epoch / Step")
        plt.ylabel("Mean Hardness")
        plt.title("Counterfactual Hardness Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "counterfactual_hardness_curve.png", dpi=300)
        plt.close()


def load_predictions(run_name="final", tta=False):
    pred_file = "tta_predictions.csv" if tta else "test_predictions.csv"
    pred_path = Path(cfg.output_dir) / "runs" / run_name / pred_file

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path)

    return (
        df["label"].values.astype(int),
        df["prob_fake"].values.astype(float),
        df["path"].tolist(),
    )


def plot_confusion_matrix(y_true, y_prob, fig_dir, fname, title):
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
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13,
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / fname, dpi=300)
    plt.close()

    return cm


def plot_roc_and_low_fpr(y_true, y_prob, fig_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.axvline(0.01, linestyle="--", label="1% FPR")
    plt.axvline(0.001, linestyle="--", label="0.1% FPR")
    plt.xlim(0, 0.05)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Low-FPR ROC Zoom")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "low_fpr_roc_zoom.png", dpi=300)
    plt.close()


def plot_score_distribution(y_true, y_prob, fig_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Real")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Fake")
    plt.xlabel("Predicted Fake Probability")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "score_distribution.png", dpi=300)
    plt.close()


def plot_reliability(y_true, y_prob, fig_dir, n_bins=15):
    y_pred = (y_prob >= cfg.threshold).astype(int)
    confidence = np.maximum(y_prob, 1 - y_prob)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    confs = []
    accs = []

    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if mask.sum() == 0:
            continue
        confs.append(confidence[mask].mean())
        accs.append(correct[mask].mean())

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.plot(confs, accs, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "reliability_diagram.png", dpi=300)
    plt.close()


def save_confusion_counts_table(y_true, y_prob, table_dir):
    y_pred = (y_prob >= cfg.threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    df = pd.DataFrame([
        {
            "True": "Real",
            "Predicted Real": int(cm[0, 0]),
            "Predicted Fake": int(cm[0, 1]),
            "Total": int(cm[0].sum()),
        },
        {
            "True": "Fake",
            "Predicted Real": int(cm[1, 0]),
            "Predicted Fake": int(cm[1, 1]),
            "Total": int(cm[1].sum()),
        },
    ])

    df.to_csv(Path(table_dir) / "confusion_matrix_counts.csv", index=False)


def save_standard_vs_tta_table(table_dir, run_name="final"):
    rows = []
    run_dir = Path(cfg.output_dir) / "runs" / run_name

    for setting, filename in [
        ("Standard", "test_metrics.json"),
        ("TTA", "tta_metrics.json"),
    ]:
        path = run_dir / filename
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            rows.append({"Setting": setting, **m})

    if rows:
        pd.DataFrame(rows).to_csv(
            Path(table_dir) / "standard_vs_tta_performance.csv",
            index=False,
        )


def failure_cases(y_true, y_prob, paths, fig_dir, max_cases=12):
    y_pred = (y_prob >= cfg.threshold).astype(int)
    errors = np.where(y_pred != y_true)[0]

    if len(errors) == 0:
        return

    conf = np.maximum(y_prob[errors], 1 - y_prob[errors])
    order = errors[np.argsort(-conf)][:max_cases]

    cols = min(4, len(order))
    rows = int(math.ceil(len(order) / cols))

    plt.figure(figsize=(cols * 3.2, rows * 3.4))

    for i, idx in enumerate(order):
        img = Image.open(paths[idx]).convert("RGB").resize((160, 160))

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)

        true_text = "Fake" if y_true[idx] == 1 else "Real"
        pred_text = "Fake" if y_pred[idx] == 1 else "Real"

        plt.title(f"T:{true_text} P:{pred_text}\nFake p={y_prob[idx]:.2f}", fontsize=9)
        plt.axis("off")

    plt.suptitle("High-Confidence Failure Cases")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "failure_case_examples.png", dpi=300)
    plt.close()


def bar_graph_from_table(csv_path, metric, fig_path, title):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        return

    label_col = "Model" if "Model" in df.columns else ("Variant" if "Variant" in df.columns else df.columns[0])

    plt.figure(figsize=(10, 5))
    plt.bar(df[label_col].astype(str), df[metric].astype(float))
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def leakage_shortcut_check(train_records, test_records, table_dir, fig_dir, max_train=20000, max_test=15000, threshold=5):
    def file_md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    train_sample = train_records[:max_train]
    test_sample = test_records[:max_test]

    train_md5 = set()
    for r in train_sample:
        try:
            train_md5.add(file_md5(r["path"]))
        except Exception:
            pass

    exact_leaks = 0

    for r in test_sample:
        try:
            if file_md5(r["path"]) in train_md5:
                exact_leaks += 1
        except Exception:
            pass

    train_hashes = []
    for r in train_sample:
        try:
            train_hashes.append((str(imagehash.phash(Image.open(r["path"]).convert("RGB"))), r["path"]))
        except Exception:
            pass

    unique_test_similar = set()

    for r in test_sample:
        try:
            h_test = imagehash.phash(Image.open(r["path"]).convert("RGB"))
            for h_tr, _p in train_hashes:
                dist = h_test - imagehash.hex_to_hash(h_tr)
                if dist <= threshold:
                    unique_test_similar.add(r["path"])
                    break
        except Exception:
            pass

    table = pd.DataFrame([{
        "Train images checked": len(train_sample),
        "Unseen images checked": len(test_sample),
        "Exact duplicate leakage": exact_leaks,
        "pHash threshold": threshold,
        "Unseen images visually similar to train": len(unique_test_similar),
        "Similarity percent": 100 * len(unique_test_similar) / max(1, len(test_sample)),
    }])

    table.to_csv(Path(table_dir) / "leakage_shortcut_check_table.csv", index=False)

    plt.figure(figsize=(8, 5))
    names = ["Exact leaks", "pHash similar unseen"]
    vals = [exact_leaks, len(unique_test_similar)]
    plt.bar(names, vals)
    plt.ylabel("Count")
    plt.title("Leakage / Shortcut Check")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "leakage_shortcut_check_graph.png", dpi=300)
    plt.close()


def gradcam_fft_explanation(lightning_module, sample_path, fig_dir):
    if lightning_module is None:
        return

    model = lightning_module.model

    if not hasattr(model, "spatial"):
        print("Grad-CAM skipped because this model has no spatial branch.")
        return

    device = next(model.parameters()).device
    model.eval()

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.activations = None
            self.gradients = None
            self.forward_handle = target_layer.register_forward_hook(self.forward_hook)
            self.backward_handle = target_layer.register_full_backward_hook(self.backward_hook)

        def forward_hook(self, module, input, output):
            self.activations = output.detach()

        def backward_hook(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def remove(self):
            self.forward_handle.remove()
            self.backward_handle.remove()

        def generate(self, x, class_idx=None):
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)
            logits = out["logits"]
            if class_idx is None:
                class_idx = int(torch.argmax(logits, dim=1).item())
            score = logits[:, class_idx].sum()
            score.backward(retain_graph=True)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().numpy(), logits

    img = Image.open(sample_path).convert("RGB")
    transform = build_eval_transform(cfg.image_size)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)["logits"]
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred = "Fake" if np.argmax(probs) == 1 else "Real"

    cam_extractor = GradCAM(model, model.spatial.layer4)
    cam, _ = cam_extractor.generate(x, class_idx=int(np.argmax(probs)))
    cam_extractor.remove()

    img_rgb = np.asarray(img.resize((cfg.image_size, cfg.image_size))).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.58 * img_rgb + 0.42 * heatmap, 0, 255).astype(np.uint8)

    gray = img.resize((cfg.image_size, cfg.image_size)).convert("L")
    arr = np.asarray(gray).astype(np.float32) / 255.0
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr))))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mag, cmap="magma")
    plt.title("FFT Map")
    plt.axis("off")

    plt.suptitle(f"{pred} | Real={probs[0]:.3f}, Fake={probs[1]:.3f}")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "gradcam_fft_explanation.png", dpi=300)
    plt.close()


def collect_project_files(bundle_dir):
    scripts_dir = bundle_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "requirements.txt",
        "config.py",
        "data.py",
        "models.py",
        "lightning_module.py",
        "report.py",
        "train.py",
        "predict_image.py",
        "external_eval.py",
        "forensic_audit.py",
        "README.md",
    ]:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, scripts_dir / name)


def make_final_zip(zip_name="fg_dg_csf_xai_research_bundle.zip"):
    output_dir = Path(cfg.output_dir)
    ckpt_dir = Path(cfg.checkpoint_dir)

    bundle_dir = Path("research_bundle")

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        shutil.copytree(output_dir, bundle_dir / "outputs", dirs_exist_ok=True)

    if ckpt_dir.exists():
        shutil.copytree(ckpt_dir, bundle_dir / "checkpoints", dirs_exist_ok=True)

    collect_project_files(bundle_dir)

    manifest_rows = []
    for p in bundle_dir.rglob("*"):
        if p.is_file():
            manifest_rows.append({
                "path": str(p.relative_to(bundle_dir)),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 4),
            })

    pd.DataFrame(manifest_rows).to_csv(bundle_dir / "MANIFEST.csv", index=False)

    readme = """
FG-DG-CSF-XAI Research Bundle

This bundle contains:
- outputs/tables/: all research tables
- outputs/figures/: all research graphs
- outputs/runs/: metrics, predictions, logs
- checkpoints/: Lightning model checkpoints
- scripts/: code used for training, evaluation, reporting, and prediction
"""

    (bundle_dir / "README_BUNDLE.txt").write_text(readme.strip(), encoding="utf-8")

    zip_path = Path(zip_name)

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", root_dir=str(bundle_dir))

    print("Created final research ZIP:", zip_path)

    return zip_path


def generate_report_outputs(run_name="final", datamodule=None, lightning_module=None):
    output_dir = Path(cfg.output_dir)
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_training_config_table(table_dir)
    create_architecture_diagram(fig_dir)
    plot_training_curves(run_name, fig_dir)

    y_true, y_prob, paths = load_predictions(run_name, tta=False)

    metrics = compute_numpy_metrics(y_true, y_prob, cfg.threshold)
    pd.DataFrame([{"Setting": "Standard", **metrics}]).to_csv(
        table_dir / "standard_performance.csv",
        index=False,
    )

    plot_confusion_matrix(y_true, y_prob, fig_dir, "confusion_matrix.png", "Confusion Matrix")
    plot_roc_and_low_fpr(y_true, y_prob, fig_dir)
    plot_score_distribution(y_true, y_prob, fig_dir)
    plot_reliability(y_true, y_prob, fig_dir, cfg.calibration_bins)
    save_confusion_counts_table(y_true, y_prob, table_dir)
    save_standard_vs_tta_table(table_dir, run_name)
    failure_cases(y_true, y_prob, paths, fig_dir)

    baseline_path = table_dir / "baseline_comparison.csv"
    ablation_path = table_dir / "ablation_study.csv"

    bar_graph_from_table(
        baseline_path,
        "auc",
        fig_dir / "baseline_comparison_bar_graph.png",
        "Baseline Comparison by AUC",
    )

    bar_graph_from_table(
        ablation_path,
        "auc",
        fig_dir / "ablation_comparison_bar_graph.png",
        "Ablation Study by AUC",
    )

    if datamodule is not None:
        leakage_shortcut_check(
            datamodule.train_records,
            datamodule.test_records,
            table_dir,
            fig_dir,
        )

    if lightning_module is not None and paths:
        gradcam_fft_explanation(
            lightning_module,
            paths[0],
            fig_dir,
        )

    return make_final_zip()
