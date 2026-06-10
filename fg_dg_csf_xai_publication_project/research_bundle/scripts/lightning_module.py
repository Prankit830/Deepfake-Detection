import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from config import cfg
from models import build_model, batch_fft_energy_vector, domain_lambda_schedule


class TensorCounterfactualAugmenter:
    """
    Lightweight tensor counterfactual augmentation.
    Kept fast so Lightning benchmark modes can run easily.
    """

    def __init__(self, p=0.85):
        self.p = p

    def __call__(self, x_norm):
        if torch.rand(1).item() > self.p:
            return x_norm

        noise_scale = float(np.random.uniform(0.005, 0.03))
        x = x_norm + torch.randn_like(x_norm) * noise_scale

        if torch.rand(1).item() < 0.35:
            x = x + float(np.random.uniform(-0.08, 0.08))

        return torch.clamp(x, -3.0, 3.0)


def tpr_at_fpr(y_true, y_prob, target_fpr):
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]

    if len(valid) == 0:
        return 0.0

    return float(np.max(tpr[valid]))


def equal_error_rate(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))

    return float((fpr[idx] + fnr[idx]) / 2.0)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])

        if mask.sum() == 0:
            continue

        ece += (mask.sum() / len(y_true)) * abs(
            correct[mask].mean() - confidence[mask].mean()
        )

    return float(ece)


def compute_numpy_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    two_classes = len(np.unique(y_true)) == 2

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if two_classes else 0.0,
        "eer": float(equal_error_rate(y_true, y_prob)) if two_classes else 0.0,
        "tpr_at_1pct_fpr": float(tpr_at_fpr(y_true, y_prob, 0.01)) if two_classes else 0.0,
        "tpr_at_0_1pct_fpr": float(tpr_at_fpr(y_true, y_prob, 0.001)) if two_classes else 0.0,
        "ece": float(expected_calibration_error(y_true, y_prob, cfg.calibration_bins)),
        "brier": float(brier_score_loss(y_true, y_prob)) if two_classes else 0.0,
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    out.update({
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    })

    return out


class DetectorLightningModule(L.LightningModule):
    def __init__(
        self,
        run_name="final",
        num_domains=2,
        model_type="proposed",
        backbone_name=None,
        use_frequency=True,
        use_patch=True,
        use_counterfactuals=True,
        use_domain=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.run_name = run_name
        self.model_type = model_type
        self.backbone_name = backbone_name
        self.use_counterfactuals = use_counterfactuals
        self.use_domain = use_domain

        self.model = build_model(
            model_type=model_type,
            num_domains=num_domains,
            dropout=cfg.dropout,
            backbone_name=backbone_name,
            use_frequency=use_frequency,
            use_patch=use_patch,
        )

        # External baselines and compact baselines do not use proposed regularizers.
        if model_type != "proposed":
            self.use_counterfactuals = False
            self.use_domain = False

        self.cf_augmenter = TensorCounterfactualAugmenter(p=cfg.cf_probability)

        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.domain_ce = nn.CrossEntropyLoss()

        self.val_probs = []
        self.val_labels = []

        self.test_probs = []
        self.test_labels = []
        self.test_paths = []

    def compute_hardness(self, logits, cf_logits, features, cf_features, x, x_cf):
        with torch.no_grad():
            p = F.softmax(logits, dim=1)[:, 1]
            p_cf = F.softmax(cf_logits, dim=1)[:, 1]

            prediction_instability = torch.abs(p - p_cf)

            freq_orig = batch_fft_energy_vector(x, bins=8)
            freq_cf = batch_fft_energy_vector(x_cf, bins=8)

            frequency_inconsistency = F.mse_loss(
                freq_orig,
                freq_cf,
                reduction="none",
            ).mean(dim=1)

            feature_shift = 1.0 - F.cosine_similarity(
                features,
                cf_features,
                dim=1,
            )

            hardness = (
                cfg.hardness_alpha * prediction_instability
                + cfg.hardness_beta * frequency_inconsistency
                + cfg.hardness_gamma * feature_shift
            )

        return hardness

    def training_step(self, batch, batch_idx):
        x, y, domain, _paths = batch

        y = y.long()
        domain = domain.long()

        grl_lambda = domain_lambda_schedule(
            self.current_epoch,
            cfg.epochs,
            cfg.domain_grl_final_lambda,
        )

        out = self.model(
            x,
            grl_lambda=grl_lambda,
            return_features=True,
        )

        logits = out["logits"]

        loss_main = self.ce(logits, y)

        loss_domain = torch.tensor(0.0, device=self.device)

        if self.use_domain:
            loss_domain = self.domain_ce(
                out["domain_logits"],
                domain,
            )

        loss_cf = torch.tensor(0.0, device=self.device)
        hardness_mean = torch.tensor(0.0, device=self.device)

        if self.use_counterfactuals:
            x_cf = self.cf_augmenter(x)

            cf_out = self.model(
                x_cf,
                grl_lambda=0.0,
                return_features=True,
            )

            cf_logits = cf_out["logits"]

            hardness = self.compute_hardness(
                logits.detach(),
                cf_logits.detach(),
                out["features"].detach(),
                cf_out["features"].detach(),
                x.detach(),
                x_cf.detach(),
            )

            hardness_mean = hardness.mean()

            k = max(
                1,
                int(math.ceil(cfg.hard_cf_fraction * x.shape[0])),
            )

            hard_idx = torch.topk(
                hardness,
                k=k,
                largest=True,
            ).indices

            cf_ce = self.ce(
                cf_logits[hard_idx],
                y[hard_idx],
            )

            consistency = F.kl_div(
                F.log_softmax(cf_logits[hard_idx], dim=1),
                F.softmax(logits.detach()[hard_idx], dim=1),
                reduction="batchmean",
            )

            loss_cf = cf_ce + cfg.cf_consistency_weight * consistency

        loss = (
            loss_main
            + cfg.cf_loss_weight * loss_cf
            + cfg.domain_loss_weight * loss_domain
        )

        probs = F.softmax(logits.detach(), dim=1)[:, 1]
        preds = (probs >= cfg.threshold).long()
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_main_loss", loss_main, on_step=False, on_epoch=True)
        self.log("train_cf_loss", loss_cf, on_step=False, on_epoch=True)
        self.log("train_domain_loss", loss_domain, on_step=False, on_epoch=True)
        self.log("train_hardness", hardness_mean, prog_bar=True, on_step=False, on_epoch=True)
        self.log("grl_lambda", grl_lambda, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _domain, _paths = batch

        y = y.long()

        out = self.model(x, grl_lambda=0.0)

        logits = out["logits"]

        loss = self.ce(logits, y)

        probs = F.softmax(logits, dim=1)[:, 1]

        self.val_probs.append(probs.detach().cpu())
        self.val_labels.append(y.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        if not self.val_probs:
            return

        probs = torch.cat(self.val_probs).numpy()
        labels = torch.cat(self.val_labels).numpy()

        metrics = compute_numpy_metrics(labels, probs, cfg.threshold)

        self.log("val_auc", metrics["auc"], prog_bar=True)
        self.log("val_acc", metrics["accuracy"], prog_bar=True)
        self.log("val_f1", metrics["f1"], prog_bar=True)

        self.val_probs.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y, _domain, paths = batch

        out = self.model(x, grl_lambda=0.0)

        logits = out["logits"]

        probs = F.softmax(logits, dim=1)[:, 1]

        self.test_probs.append(probs.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        self.test_paths.extend(list(paths))

    def on_test_epoch_end(self):
        probs = torch.cat(self.test_probs).numpy()
        labels = torch.cat(self.test_labels).numpy()

        metrics = compute_numpy_metrics(labels, probs, cfg.threshold)

        run_dir = Path(cfg.output_dir) / "runs" / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        import pandas as pd

        pd.DataFrame({
            "path": self.test_paths,
            "label": labels.astype(int),
            "prob_fake": probs.astype(float),
        }).to_csv(run_dir / "test_predictions.csv", index=False)

        print(f"TEST METRICS [{self.run_name}]:", metrics)

        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                self.log(f"test_{k}", v)

        self.test_probs.clear()
        self.test_labels.clear()
        self.test_paths.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.epochs),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
