import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.0):
        super().__init__()
        init_temperature = float(init_temperature)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temperature], dtype=torch.float32)))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.05, 20.0)

    def forward(self, logits):
        return logits / self.temperature


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi)

        if mask.sum() == 0:
            continue

        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - confidence[mask].mean())

    return float(ece)


def fit_temperature(logits, labels, max_iter=80, device="cpu"):
    logits = logits.detach().to(device)
    labels = labels.detach().long().to(device)

    scaler = TemperatureScaler(init_temperature=1.0).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [scaler.log_temperature],
        lr=0.05,
        max_iter=max_iter
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return scaler


def load_temperature(path):
    path = Path(path)

    if not path.exists():
        return 1.0

    with open(path, "r") as f:
        payload = json.load(f)

    return float(payload.get("temperature", 1.0))


def load_calibration_metrics(path):
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def save_temperature_artifacts(output_dir, temperature, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_path = output_dir / "temperature.json"
    metrics_path = output_dir / "calibration_metrics.json"

    with open(temperature_path, "w") as f:
        json.dump(
            {
                "temperature": float(temperature),
                **metrics,
            },
            f,
            indent=2,
        )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return temperature_path, metrics_path


def compute_calibration_report(logits_before, labels, temperature, n_bins=15):
    logits_before = logits_before.detach().cpu()
    labels_np = labels.detach().cpu().numpy().astype(int)

    probs_before = F.softmax(logits_before, dim=1)[:, 1].numpy()

    logits_after = logits_before / float(temperature)
    probs_after = F.softmax(logits_after, dim=1)[:, 1].numpy()

    report = {
        "temperature": float(temperature),
        "n_samples": int(len(labels_np)),
        "ece_before_temperature": expected_calibration_error(labels_np, probs_before, n_bins=n_bins),
        "ece_after_temperature": expected_calibration_error(labels_np, probs_after, n_bins=n_bins),
        "brier_before_temperature": float(brier_score_loss(labels_np, probs_before)),
        "brier_after_temperature": float(brier_score_loss(labels_np, probs_after)),
    }

    # Convenience keys used by the web app
    report["ece"] = report["ece_after_temperature"]
    report["brier"] = report["brier_after_temperature"]

    return report
