import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

from config import cfg
from data import build_eval_transform
from train import load_lightning_model


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


def overlay_cam(img_rgb, cam, alpha=0.42):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return np.clip((1 - alpha) * img_rgb + alpha * heatmap, 0, 255).astype(np.uint8)


def frequency_heatmap(img, image_size):
    img = img.resize((image_size, image_size)).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(fft))

    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)


def explain_image(lit_model, image_path, save_path):
    device = next(lit_model.model.parameters()).device
    model = lit_model.model
    model.eval()

    img = Image.open(image_path).convert("RGB")

    transform = build_eval_transform(cfg.image_size)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)["logits"]
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = "Fake" if pred_idx == 1 else "Real"

    img_rgb = np.asarray(img.resize((cfg.image_size, cfg.image_size))).astype(np.uint8)
    freq_map = frequency_heatmap(img, cfg.image_size)

    if hasattr(model, "spatial"):
        gradcam = GradCAM(model, model.spatial.layer4)
        cam, _ = gradcam.generate(x, class_idx=pred_idx)
        gradcam.remove()
        overlay = overlay_cam(img_rgb, cam)
    else:
        overlay = img_rgb

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM" if hasattr(model, "spatial") else "Model Evidence")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(freq_map, cmap="magma")
    plt.title("FFT Map")
    plt.axis("off")

    plt.suptitle(f"{pred_label} | Real={probs[0]:.3f}, Fake={probs[1]:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Prediction:", pred_label)
    print("Real probability:", float(probs[0]))
    print("Fake probability:", float(probs[1]))
    print("Saved explanation:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--run_name", default="final")
    parser.add_argument("--save_path", default="outputs/external_prediction.png")
    args = parser.parse_args()

    lit_model = load_lightning_model(args.run_name)

    if lit_model is None:
        raise FileNotFoundError(f"No checkpoint found for run: {args.run_name}")

    explain_image(lit_model, args.image, args.save_path)


if __name__ == "__main__":
    main()
