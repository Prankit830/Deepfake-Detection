import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from config import cfg
from data import DeepfakeDataModule
from lightning_module import DetectorLightningModule, compute_numpy_metrics
from report import generate_report_outputs


VARIANTS = {
    # Frequency-oriented baselines
    "freqnet_like_baseline": dict(
        model_type="freqnet_like",
        backbone_name=None,
        use_frequency=True,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "f3net_like_baseline": dict(
        model_type="f3net_like",
        backbone_name=None,
        use_frequency=True,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),

    # Compact detector
    "mesonet_baseline": dict(
        model_type="mesonet",
        backbone_name=None,
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),

    # Strong modern CNN / transformer baselines
    "efficientnet_b4_baseline": dict(
        model_type="timm",
        backbone_name="efficientnet_b4",
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "convnext_tiny_baseline": dict(
        model_type="timm",
        backbone_name="convnext_tiny",
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "vit_b16_baseline": dict(
        model_type="timm",
        backbone_name="vit_base_patch16_224",
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "swin_tiny_baseline": dict(
        model_type="timm",
        backbone_name="swin_tiny_patch4_window7_224",
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "xception_baseline": dict(
        model_type="timm",
        backbone_name="xception",
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),

    # Internal ablations
    "spatial_baseline": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=False,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "spatial_frequency": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=False,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "spatial_frequency_patch": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=True,
        use_counterfactuals=False,
        use_domain=False,
    ),
    "no_counterfactual": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=True,
        use_counterfactuals=False,
        use_domain=True,
    ),
    "no_domain": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=True,
        use_counterfactuals=True,
        use_domain=False,
    ),

    # Full proposed model
    "final": dict(
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=True,
        use_counterfactuals=True,
        use_domain=True,
    ),
}


PAPER_VARIANTS = [
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


def get_checkpoint_dir(run_name):
    ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def find_best_or_last_checkpoint(run_name):
    ckpt_dir = get_checkpoint_dir(run_name)
    best_ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))

    if best_ckpts:
        return best_ckpts[-1]

    last_ckpt = ckpt_dir / "last.ckpt"

    if last_ckpt.exists():
        return last_ckpt

    return None


def run_one_experiment(run_name, datamodule, max_epochs, force=False):
    cfg.epochs = max_epochs
    flags = VARIANTS[run_name]

    metrics_path = Path(cfg.output_dir) / "runs" / run_name / "test_metrics.json"

    if metrics_path.exists() and not force:
        print(f"Skipping {run_name}; metrics already exist: {metrics_path}")
        return None, find_best_or_last_checkpoint(run_name)

    lit_model = DetectorLightningModule(
        run_name=run_name,
        num_domains=2,
        **flags,
    )

    ckpt_dir = get_checkpoint_dir(run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = CSVLogger(
        save_dir=Path(cfg.output_dir) / "runs" / run_name,
        name="logs",
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=20,
        default_root_dir=str(Path(cfg.output_dir) / "runs" / run_name),
    )

    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() and not force else None

    if ckpt_path:
        print(f"Resuming {run_name} from {ckpt_path}")
    else:
        print(f"Starting {run_name} for {max_epochs} epoch(s)")

    trainer.fit(
        lit_model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    trainer.test(
        lit_model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    return lit_model, checkpoint_callback.best_model_path


@torch.no_grad()
def run_tta(lit_model, datamodule, run_name="final", rounds=None):
    if lit_model is None:
        lit_model = load_lightning_model(run_name)

    if lit_model is None:
        print("TTA skipped because model could not be loaded.")
        return None

    rounds = cfg.tta_rounds if rounds is None else rounds

    model = lit_model.model
    model.eval()

    device = lit_model.device

    y_true = []
    y_prob = []
    paths = []

    for x, y, _domain, batch_paths in datamodule.test_dataloader():
        x = x.to(device)

        probs_all = []

        for _ in range(rounds):
            logits = model(x)["logits"]
            probs_all.append(F.softmax(logits, dim=1)[:, 1])

            x_flip = torch.flip(x, dims=[3])
            logits_flip = model(x_flip)["logits"]
            probs_all.append(F.softmax(logits_flip, dim=1)[:, 1])

        prob = torch.stack(probs_all, dim=0).mean(dim=0)

        y_prob.extend(prob.detach().cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
        paths.extend(list(batch_paths))

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    metrics = compute_numpy_metrics(y_true, y_prob, cfg.threshold)

    run_dir = Path(cfg.output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "tta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "path": paths,
        "label": y_true,
        "prob_fake": y_prob,
    }).to_csv(run_dir / "tta_predictions.csv", index=False)

    print("TTA metrics:", metrics)

    return metrics


def write_comparison_tables(rows):
    table_dir = Path(cfg.output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return

    df.to_csv(table_dir / "ablation_study.csv", index=False)

    baseline_names = [
        "freqnet_like_baseline",
        "f3net_like_baseline",
        "mesonet_baseline",
        "efficientnet_b4_baseline",
        "convnext_tiny_baseline",
        "vit_b16_baseline",
        "swin_tiny_baseline",
        "xception_baseline",
        "spatial_baseline",
        "final",
    ]

    baseline = df[df["Variant"].isin(baseline_names)].copy()

    name_map = {
        "freqnet_like_baseline": "FreqNet-like frequency baseline",
        "f3net_like_baseline": "F3Net-like spatial-frequency baseline",
        "mesonet_baseline": "MesoNet compact baseline",
        "efficientnet_b4_baseline": "EfficientNet-B4 CNN baseline",
        "convnext_tiny_baseline": "ConvNeXt-Tiny CNN baseline",
        "vit_b16_baseline": "ViT-B/16 transformer baseline",
        "swin_tiny_baseline": "Swin-Tiny transformer baseline",
        "xception_baseline": "Xception baseline",
        "spatial_baseline": "Spatial-only internal baseline",
        "final": "Full proposed FG-DG-CSF-XAI",
    }

    baseline["Model"] = baseline["Variant"].map(name_map)

    cols = ["Model", "Variant"] + [
        c for c in baseline.columns
        if c not in ["Model", "Variant"]
    ]

    baseline[cols].to_csv(table_dir / "baseline_comparison.csv", index=False)


def collect_existing_metrics():
    rows = []

    for name in VARIANTS.keys():
        metrics_path = Path(cfg.output_dir) / "runs" / name / "test_metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)

            rows.append({"Variant": name, **m})

    return rows


def load_lightning_model(run_name="final"):
    ckpt_path = find_best_or_last_checkpoint(run_name)

    if ckpt_path is None:
        print(f"No checkpoint found for {run_name}.")
        return None

    print(f"Loading {run_name} from {ckpt_path}")

    lit_model = DetectorLightningModule.load_from_checkpoint(
        str(ckpt_path),
        strict=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)
    lit_model.eval()

    return lit_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["final", "paper", "all", "report"],
        default="final",
    )

    parser.add_argument(
        "--final_epochs",
        type=int,
        default=cfg.final_epochs,
        help="Epochs for full proposed model. Default: 12.",
    )

    parser.add_argument(
        "--paper_epochs",
        type=int,
        default=cfg.paper_epochs,
        help="Epochs for baseline/ablation/paper runs. Default: 3.",
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Run TTA evaluation for final model.",
    )

    parser.add_argument(
        "--leakage_check",
        action="store_true",
        help="Run pHash/shortcut leakage check in report generation.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if metrics already exist.",
    )

    args = parser.parse_args()

    L.seed_everything(cfg.seed, workers=True)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    datamodule = DeepfakeDataModule()
    datamodule.setup()

    final_model = None

    if args.mode in ["final", "all"]:
        final_model, _ = run_one_experiment(
            "final",
            datamodule,
            max_epochs=args.final_epochs,
            force=args.force,
        )

        if final_model is None:
            final_model = load_lightning_model("final")

        if args.tta:
            run_tta(final_model, datamodule, "final")

        generate_report_outputs(
            "final",
            datamodule=datamodule if args.leakage_check else None,
            lightning_module=final_model,
        )

    if args.mode in ["paper", "all"]:
        rows = collect_existing_metrics()

        for name in PAPER_VARIANTS:
            metrics_path = Path(cfg.output_dir) / "runs" / name / "test_metrics.json"

            if metrics_path.exists() and not args.force:
                print(f"Skipping {name}; metrics already exist.")
                continue

            run_one_experiment(
                name,
                datamodule,
                max_epochs=args.paper_epochs,
                force=args.force,
            )

        rows = collect_existing_metrics()
        write_comparison_tables(rows)

        if final_model is None:
            final_model = load_lightning_model("final")

        generate_report_outputs(
            "final",
            datamodule=datamodule if args.leakage_check else None,
            lightning_module=final_model,
        )

    if args.mode == "report":
        rows = collect_existing_metrics()

        if rows:
            write_comparison_tables(rows)

        final_model = load_lightning_model("final")

        generate_report_outputs(
            "final",
            datamodule=datamodule if args.leakage_check else None,
            lightning_module=final_model,
        )


if __name__ == "__main__":
    main()
