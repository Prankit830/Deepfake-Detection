import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make project root importable when this file is inside web_app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from data import DeepfakeDataModule
from lightning_module import DetectorLightningModule
from web_app.calibration import fit_temperature, compute_calibration_report, save_temperature_artifacts


@torch.no_grad()
def collect_logits_and_labels(lit_model, loader, device):
    lit_model.eval()
    model = lit_model.model
    model.eval()

    logits_all = []
    labels_all = []

    for x, y, _domain, _paths in loader:
        x = x.to(device)
        logits = model(x)["logits"].detach().cpu()
        logits_all.append(logits)
        labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0).long()

    return logits_all, labels_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final/last.ckpt")
    parser.add_argument("--output_dir", default="web_app/artifacts")
    parser.add_argument("--max_iter", type=int, default=80)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = DeepfakeDataModule()
    dm.setup()

    lit_model = DetectorLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        strict=True,
        map_location=device,
    )
    lit_model.to(device)
    lit_model.eval()

    logits, labels = collect_logits_and_labels(
        lit_model,
        dm.val_dataloader(),
        device,
    )

    scaler = fit_temperature(
        logits.to(device),
        labels.to(device),
        max_iter=args.max_iter,
        device=device,
    )

    temperature = float(scaler.temperature.detach().cpu().item())

    report = compute_calibration_report(
        logits,
        labels,
        temperature,
        n_bins=cfg.calibration_bins,
    )

    temp_path, metrics_path = save_temperature_artifacts(
        args.output_dir,
        temperature,
        report,
    )

    print("Learned temperature:", temperature)
    print("Saved:", temp_path)
    print("Saved:", metrics_path)
    print(report)


if __name__ == "__main__":
    main()
