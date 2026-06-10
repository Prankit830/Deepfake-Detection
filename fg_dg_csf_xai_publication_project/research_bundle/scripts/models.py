import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)

from data import IMAGENET_MEAN, IMAGENET_STD


MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD_TENSOR = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def denormalize_batch(x):
    mean = MEAN_TENSOR.to(device=x.device, dtype=x.dtype)
    std = STD_TENSOR.to(device=x.device, dtype=x.dtype)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def rgb_to_gray(x01):
    return (
        0.2989 * x01[:, 0:1]
        + 0.5870 * x01[:, 1:2]
        + 0.1140 * x01[:, 2:3]
    )


def make_fft_map(x_norm, scale=1.0):
    """
    AMP-safe FFT map.
    FFT is forced to float32.
    """
    device_type = "cuda" if x_norm.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        x01 = denormalize_batch(x_norm.float())
        gray = rgb_to_gray(x01)

        if scale != 1.0:
            size = (
                max(16, int(gray.shape[-2] * scale)),
                max(16, int(gray.shape[-1] * scale)),
            )
            gray = F.interpolate(gray, size=size, mode="bilinear", align_corners=False)

        fft = torch.fft.fft2(gray.squeeze(1).float(), norm="ortho")
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft)).unsqueeze(1)

        flat = mag.flatten(1)
        mn = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        mx = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        mag = (mag - mn) / (mx - mn + 1e-6)

        return mag.repeat(1, 3, 1, 1)


def batch_fft_energy_vector(x_norm, bins=8):
    device_type = "cuda" if x_norm.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        x01 = denormalize_batch(x_norm.float())
        gray = rgb_to_gray(x01).squeeze(1)
        fft = torch.fft.fftshift(torch.fft.fft2(gray.float(), norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))

        h, w = mag.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=mag.device),
            torch.linspace(-1, 1, w, device=mag.device),
            indexing="ij",
        )
        radius = torch.sqrt(xx**2 + yy**2)
        edges = torch.linspace(0, radius.max().item() + 1e-6, bins + 1, device=mag.device)

        energies = []
        for i in range(bins):
            mask = (radius >= edges[i]) & (radius < edges[i + 1])
            energies.append(mag[:, mask].mean(dim=1))

        return F.normalize(torch.stack(energies, dim=1), dim=1)


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradientReversalFn.apply(x, lambd)


def domain_lambda_schedule(epoch, total_epochs, final_lambda):
    if total_epochs <= 1:
        return final_lambda
    p = epoch / max(1, total_epochs - 1)
    return float(final_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


class SpatialBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmap = self.layer4(x)
        vec = self.avgpool(fmap).flatten(1)
        return vec, fmap


class ResNet18FeatureEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


class MultiScaleFFTBranch(nn.Module):
    def __init__(self, scales=(1.0, 0.5, 0.25), pretrained=True):
        super().__init__()
        self.scales = scales
        self.encoder = ResNet18FeatureEncoder(pretrained=pretrained)
        self.attn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x_norm):
        feats = []
        maps = []

        for scale in self.scales:
            fft_map = make_fft_map(x_norm, scale=scale)
            if fft_map.shape[-2:] != x_norm.shape[-2:]:
                fft_map = F.interpolate(fft_map, size=x_norm.shape[-2:], mode="bilinear", align_corners=False)
            maps.append(fft_map)
            feats.append(self.encoder(fft_map))

        stacked = torch.stack(feats, dim=1)
        weights = torch.softmax(self.attn(stacked).squeeze(-1), dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, maps, weights


class PatchFrequencyDescriptor(nn.Module):
    def __init__(self, patch_size=16, bands=6, out_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.bands = bands
        self.mlp = nn.Sequential(
            nn.Linear(bands * 2, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_norm):
        device_type = "cuda" if x_norm.is_cuda else "cpu"

        with torch.amp.autocast(device_type=device_type, enabled=False):
            x01 = denormalize_batch(x_norm.float())
            gray = rgb_to_gray(x01)

            p = self.patch_size
            h = (gray.shape[-2] // p) * p
            w = (gray.shape[-1] // p) * p
            gray = gray[:, :, :h, :w]

            patches = F.unfold(gray, kernel_size=p, stride=p).transpose(1, 2)
            patches = patches.reshape(gray.shape[0], -1, p, p)

            freq = torch.log1p(torch.abs(torch.fft.rfft2(patches.float(), norm="ortho")))
            fh, fw = freq.shape[-2:]

            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, fh, device=freq.device),
                torch.linspace(0, 1, fw, device=freq.device),
                indexing="ij",
            )
            radius = torch.sqrt(xx**2 + yy**2)
            edges = torch.linspace(0, radius.max().item() + 1e-6, self.bands + 1, device=freq.device)

            band_feats = []
            for i in range(self.bands):
                mask = (radius >= edges[i]) & (radius < edges[i + 1])
                band_feats.append(freq[:, :, mask].mean(dim=2))

            per_patch = torch.stack(band_feats, dim=2)
            mean = per_patch.mean(dim=1)
            std = per_patch.std(dim=1)
            feat = torch.cat([mean, std], dim=1)

        return self.mlp(feat)


class ProposedModel(nn.Module):
    """
    Final proposed FG-DG-CSF-XAI model.
    """

    def __init__(
        self,
        num_classes=2,
        num_domains=2,
        pretrained=True,
        dropout=0.45,
        use_frequency=True,
        use_patch=True,
    ):
        super().__init__()
        self.use_frequency = use_frequency
        self.use_patch = use_patch

        self.spatial = SpatialBackbone(pretrained=pretrained)
        self.frequency = MultiScaleFFTBranch(pretrained=pretrained)
        self.patch_frequency = PatchFrequencyDescriptor(out_dim=128)

        fusion_dim = 512
        self.spatial_proj = nn.Sequential(
            nn.Linear(2048, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(512, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(128, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.65),
            nn.Linear(256, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_domains),
        )

    def forward(self, x, grl_lambda=0.0, return_features=False):
        spatial_vec, spatial_map = self.spatial(x)
        s = self.spatial_proj(spatial_vec)

        if self.use_frequency:
            freq_vec, _fft_maps, scale_weights = self.frequency(x)
            f = self.freq_proj(freq_vec)
        else:
            scale_weights = None
            f = torch.zeros_like(s)

        if self.use_patch:
            patch_vec = self.patch_frequency(x)
            p = self.patch_proj(patch_vec)
        else:
            p = torch.zeros_like(s)

        gate = self.gate(torch.cat([s, f, p], dim=1))
        fused = self.fusion_norm(gate * s + (1.0 - gate) * f + 0.25 * p)
        logits = self.classifier(fused)
        domain_logits = self.domain_classifier(grad_reverse(fused, grl_lambda))

        out = {
            "logits": logits,
            "domain_logits": domain_logits,
            "features": fused,
            "spatial_map": spatial_map,
            "fft_scale_weights": scale_weights,
            "gate_mean": gate.mean(dim=1),
        }

        if return_features:
            out["spatial_features"] = s
            out["frequency_features"] = f
            out["patch_features"] = p

        return out


class MesoNetBaseline(nn.Module):
    def __init__(self, num_classes=2, num_domains=2, dropout=0.45):
        super().__init__()
        self.num_domains = num_domains
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 8, 5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, grl_lambda=0.0, return_features=False):
        fmap = self.features(x)
        features = fmap.flatten(1)
        logits = self.classifier(fmap)
        domain_logits = torch.zeros(
            features.shape[0],
            self.num_domains,
            device=features.device,
            dtype=logits.dtype,
        )
        return {"logits": logits, "domain_logits": domain_logits, "features": features}


class FreqNetLikeBaseline(nn.Module):
    """
    Frequency-only baseline. This is a reproducible frequency baseline,
    not an official FreqNet reproduction.
    """

    def __init__(self, num_classes=2, num_domains=2, pretrained=True, dropout=0.45):
        super().__init__()
        self.num_domains = num_domains
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.65),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, grl_lambda=0.0, return_features=False):
        fft_map = make_fft_map(x)
        features = self.pool(self.encoder(fft_map)).flatten(1)
        logits = self.classifier(features)
        domain_logits = torch.zeros(
            features.shape[0],
            self.num_domains,
            device=features.device,
            dtype=logits.dtype,
        )
        return {"logits": logits, "domain_logits": domain_logits, "features": features}


class F3NetLikeBaseline(nn.Module):
    """
    Spatial + frequency baseline. This is a reproducible F3Net-like baseline,
    not an official F3-Net reproduction.
    """

    def __init__(self, num_classes=2, num_domains=2, pretrained=True, dropout=0.45):
        super().__init__()
        self.num_domains = num_domains
        self.spatial = SpatialBackbone(pretrained=pretrained)
        self.frequency = MultiScaleFFTBranch(pretrained=pretrained)
        self.s_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.LayerNorm(512))
        self.f_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.LayerNorm(512))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(dropout * 0.65),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, grl_lambda=0.0, return_features=False):
        sv, _ = self.spatial(x)
        fv, _, _ = self.frequency(x)
        s = self.s_proj(sv)
        f = self.f_proj(fv)
        features = torch.cat([s, f], dim=1)
        logits = self.classifier(features)
        domain_logits = torch.zeros(
            features.shape[0],
            self.num_domains,
            device=features.device,
            dtype=logits.dtype,
        )
        return {"logits": logits, "domain_logits": domain_logits, "features": features}


class TimmBaselineModel(nn.Module):
    """
    Strong external baseline via timm:
    efficientnet_b4, convnext_tiny, vit_base_patch16_224,
    swin_tiny_patch4_window7_224, xception / legacy_xception, etc.
    """

    def __init__(
        self,
        backbone_name="efficientnet_b4",
        num_classes=2,
        num_domains=2,
        pretrained=True,
        dropout=0.45,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Install timm first: pip install timm") from e

        self.num_domains = num_domains
        self.backbone_name = backbone_name

        candidates = [backbone_name]
        if backbone_name == "xception":
            candidates = ["legacy_xception", "xception"] + timm.list_models("*xception*")

        last_error = None

        for candidate in candidates:
            try:
                self.backbone = timm.create_model(
                    candidate,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                self.backbone_name = candidate
                in_features = self.backbone.num_features
                break
            except Exception as e:
                last_error = e
        else:
            raise RuntimeError(
                f"Could not create timm backbone '{backbone_name}'. Last error: {last_error}"
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.65),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, grl_lambda=0.0, return_features=False):
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim > 2:
            features = features.flatten(1)

        logits = self.classifier(features)
        domain_logits = torch.zeros(
            features.shape[0],
            self.num_domains,
            device=features.device,
            dtype=logits.dtype,
        )

        return {"logits": logits, "domain_logits": domain_logits, "features": features}


def build_model(
    model_type,
    num_domains=2,
    dropout=0.45,
    backbone_name=None,
    use_frequency=True,
    use_patch=True,
):
    if model_type == "proposed":
        return ProposedModel(
            num_classes=2,
            num_domains=num_domains,
            pretrained=True,
            dropout=dropout,
            use_frequency=use_frequency,
            use_patch=use_patch,
        )

    if model_type == "mesonet":
        return MesoNetBaseline(num_classes=2, num_domains=num_domains, dropout=dropout)

    if model_type == "freqnet_like":
        return FreqNetLikeBaseline(num_classes=2, num_domains=num_domains, pretrained=True, dropout=dropout)

    if model_type == "f3net_like":
        return F3NetLikeBaseline(num_classes=2, num_domains=num_domains, pretrained=True, dropout=dropout)

    if model_type == "timm":
        return TimmBaselineModel(
            backbone_name=backbone_name,
            num_classes=2,
            num_domains=num_domains,
            pretrained=True,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model_type: {model_type}")
