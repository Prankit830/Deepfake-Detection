import json
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Make project root importable when this file is inside web_app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from data import build_eval_transform
from lightning_module import DetectorLightningModule
from web_app.calibration import load_temperature, load_calibration_metrics


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
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = out["logits"]

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM failed: activations or gradients missing.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), logits


def find_default_checkpoint():
    candidates = [
        PROJECT_ROOT / "checkpoints" / "final" / "last.ckpt",
    ]

    best_candidates = sorted((PROJECT_ROOT / "checkpoints" / "final").glob("best-*.ckpt")) if (PROJECT_ROOT / "checkpoints" / "final").exists() else []
    candidates.extend(best_candidates[::-1])

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def overlay_cam_on_image(img_rgb, cam, alpha=0.42):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1.0 - alpha) * img_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    return overlay


def frequency_heatmap_from_pil(img, image_size):
    img = img.resize((image_size, image_size)).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0

    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(fft))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    return mag


def cam_region_summary(cam):
    h, w = cam.shape

    threshold = np.percentile(cam, 85)
    ys, xs = np.where(cam >= threshold)

    if len(xs) == 0:
        return "central image regions"

    y_mean = ys.mean() / h
    x_mean = xs.mean() / w

    if y_mean < 0.33:
        vertical = "upper region"
    elif y_mean > 0.66:
        vertical = "lower region"
    else:
        vertical = "central region"

    if x_mean < 0.33:
        horizontal = "left side"
    elif x_mean > 0.66:
        horizontal = "right side"
    else:
        horizontal = "center"

    return f"{vertical}, concentrated near the {horizontal}"


def compute_frequency_cues(freq_map):
    h, w = freq_map.shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h),
        np.linspace(-1, 1, w),
        indexing="ij",
    )

    radius = np.sqrt(xx ** 2 + yy ** 2)

    low_mask = radius < 0.25
    mid_mask = (radius >= 0.25) & (radius < 0.60)
    high_mask = radius >= 0.60

    low_energy = float(freq_map[low_mask].mean())
    mid_energy = float(freq_map[mid_mask].mean())
    high_energy = float(freq_map[high_mask].mean())

    return {
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "high_to_low": high_energy / (low_energy + 1e-8),
        "mid_to_low": mid_energy / (low_energy + 1e-8),
    }


def compute_spatial_cues(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    edges = cv2.Canny(gray, threshold1=80, threshold2=160)

    return {
        "edge_density": float((edges > 0).mean()),
        "brightness": float(gray_float.mean()),
        "contrast": float(gray_float.std()),
    }


def describe_cues(freq_cues, spatial_cues):
    phrases = []

    high_to_low = freq_cues["high_to_low"]
    mid_to_low = freq_cues["mid_to_low"]
    edge_density = spatial_cues["edge_density"]
    contrast = spatial_cues["contrast"]
    brightness = spatial_cues["brightness"]

    if high_to_low > 0.75:
        phrases.append("noticeable high-frequency energy that may indicate synthetic texture or sharpening artifacts")
    elif high_to_low < 0.35:
        phrases.append("smooth frequency behavior with limited high-frequency disturbance")
    else:
        phrases.append("moderate high-frequency structure")

    if mid_to_low > 0.65:
        phrases.append("mid-frequency patterns that may reflect repeated or generated texture")

    if edge_density > 0.18:
        phrases.append("dense edge patterns around the highlighted region")
    elif edge_density < 0.06:
        phrases.append("smooth regions with relatively few sharp edges")
    else:
        phrases.append("balanced edge structure")

    if contrast > 0.26:
        phrases.append("strong local contrast variation")
    elif contrast < 0.12:
        phrases.append("low local contrast and soft texture")

    if brightness > 0.72:
        phrases.append("bright regions that can influence artifact visibility")
    elif brightness < 0.25:
        phrases.append("dark regions where subtle artifacts may be harder to detect")

    return phrases


def save_combined_visualization(img_rgb, overlay, freq_map, save_path, title):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(freq_map, cmap="magma")
    plt.title("FFT Magnitude Map")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_image(path, arr, cmap=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


class WebDetector:
    def __init__(
        self,
        checkpoint_path=None,
        temperature_path=None,
        calibration_metrics_path=None,
        static_output_dir=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else find_default_checkpoint()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Train the final model first or set MODEL_CKPT=/path/to/checkpoint.ckpt"
            )

        self.lit_model = DetectorLightningModule.load_from_checkpoint(
            str(self.checkpoint_path),
            strict=True,
            map_location=self.device,
        )
        self.lit_model.to(self.device)
        self.lit_model.eval()

        self.model = self.lit_model.model
        self.model.eval()

        self.temperature_path = Path(temperature_path) if temperature_path else PROJECT_ROOT / "web_app" / "artifacts" / "temperature.json"
        self.calibration_metrics_path = Path(calibration_metrics_path) if calibration_metrics_path else PROJECT_ROOT / "web_app" / "artifacts" / "calibration_metrics.json"

        self.temperature = load_temperature(self.temperature_path)
        self.calibration_metrics = load_calibration_metrics(self.calibration_metrics_path)

        self.static_output_dir = Path(static_output_dir) if static_output_dir else PROJECT_ROOT / "web_app" / "static" / "results"
        self.static_output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = build_eval_transform(cfg.image_size)

    def reload_temperature(self):
        self.temperature = load_temperature(self.temperature_path)
        self.calibration_metrics = load_calibration_metrics(self.calibration_metrics_path)

    def predict(self, image_path, known_label=None):
        image_path = Path(image_path)

        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((cfg.image_size, cfg.image_size))
        img_rgb = np.asarray(img_resized).astype(np.uint8)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logits = self.model(x)["logits"]
            calibrated_logits = raw_logits / float(self.temperature)
            calibrated_probs = F.softmax(calibrated_logits, dim=1)[0].detach().cpu().numpy()
            raw_probs = F.softmax(raw_logits, dim=1)[0].detach().cpu().numpy()

        real_prob = float(calibrated_probs[0])
        fake_prob = float(calibrated_probs[1])
        pred_idx = int(np.argmax(calibrated_probs))
        pred_label = "Fake" if pred_idx == 1 else "Real"
        confidence = max(real_prob, fake_prob)

        raw_logits_list = [float(v) for v in raw_logits[0].detach().cpu().numpy().tolist()]
        calibrated_logits_list = [float(v) for v in calibrated_logits[0].detach().cpu().numpy().tolist()]

        gradcam_available = hasattr(self.model, "spatial") and hasattr(self.model.spatial, "layer4")

        if gradcam_available:
            gradcam = GradCAM(self.model, self.model.spatial.layer4)
            cam, _ = gradcam.generate(x, class_idx=pred_idx)
            gradcam.remove()
            overlay = overlay_cam_on_image(img_rgb, cam)
            region_text = cam_region_summary(cam)
        else:
            cam = np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32)
            overlay = img_rgb
            region_text = "the model backbone does not expose the proposed spatial layer for Grad-CAM"

        freq_map = frequency_heatmap_from_pil(img, cfg.image_size)
        freq_cues = compute_frequency_cues(freq_map)
        spatial_cues = compute_spatial_cues(img_rgb)
        cue_phrases = describe_cues(freq_cues, spatial_cues)
        cue_text = "; ".join(cue_phrases[:4])

        if pred_label == "Fake":
            explanation = (
                f"The model predicts FAKE with confidence {confidence:.3f}. "
                f"The strongest visual evidence is around {region_text}. "
                f"The image cues include {cue_text}. "
                "These cues may indicate synthetic texture, abnormal local structure, "
                "or frequency-domain generator traces."
            )
        else:
            explanation = (
                f"The model predicts REAL with confidence {confidence:.3f}. "
                f"The strongest visual evidence is around {region_text}. "
                f"The observed cues include {cue_text}. "
                "The model did not find strong fake-generation evidence in the highlighted region."
            )

        job_id = uuid.uuid4().hex[:12]
        result_dir = self.static_output_dir / job_id
        result_dir.mkdir(parents=True, exist_ok=True)

        input_out = result_dir / "input.png"
        overlay_out = result_dir / "gradcam_overlay.png"
        fft_out = result_dir / "fft_magnitude.png"
        combined_out = result_dir / "combined_explanation.png"

        Image.fromarray(img_rgb).save(input_out)
        Image.fromarray(overlay).save(overlay_out)
        save_image(fft_out, freq_map, cmap="magma")
        save_combined_visualization(
            img_rgb,
            overlay,
            freq_map,
            combined_out,
            f"{pred_label} | Real={real_prob:.3f}, Fake={fake_prob:.3f}",
        )

        known_label_clean = None
        single_image_brier = None
        single_image_calibration_error = None

        if known_label is not None and str(known_label).lower() in ["real", "fake"]:
            known_label_clean = str(known_label).lower()
            y = 0 if known_label_clean == "real" else 1
            single_image_brier = float((fake_prob - y) ** 2)

            correct = 1.0 if pred_idx == y else 0.0
            single_image_calibration_error = float(abs(confidence - correct))

        model_ece = self.calibration_metrics.get("ece", self.calibration_metrics.get("ece_after_temperature"))
        model_brier = self.calibration_metrics.get("brier", self.calibration_metrics.get("brier_after_temperature"))

        result = {
            "job_id": job_id,
            "prediction": pred_label,
            "confidence": confidence,
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "raw_real_probability": float(raw_probs[0]),
            "raw_fake_probability": float(raw_probs[1]),
            "temperature": float(self.temperature),
            "raw_logits": raw_logits_list,
            "calibrated_logits": calibrated_logits_list,
            "model_ece": None if model_ece is None else float(model_ece),
            "model_brier": None if model_brier is None else float(model_brier),
            "single_image_brier": single_image_brier,
            "single_image_calibration_error": single_image_calibration_error,
            "known_label": known_label_clean,
            "ece_note": "ECE is a dataset-level metric. The app displays model-level ECE from validation/test calibration if available.",
            "brier_note": "Single-image Brier is shown only when the true label is provided.",
            "region_text": region_text,
            "frequency_cues": freq_cues,
            "spatial_cues": spatial_cues,
            "cue_phrases": cue_phrases,
            "explanation": explanation,
            "input_image_url": f"/static/results/{job_id}/input.png",
            "gradcam_url": f"/static/results/{job_id}/gradcam_overlay.png",
            "fft_url": f"/static/results/{job_id}/fft_magnitude.png",
            "combined_url": f"/static/results/{job_id}/combined_explanation.png",
            "checkpoint": str(self.checkpoint_path),
            "calibration_metrics": self.calibration_metrics,
        }

        with open(result_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        return result
