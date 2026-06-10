import io
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import lightning as L

from config import cfg


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

REAL_ALIASES = {
    "real", "reals", "authentic", "original", "originals", "pristine",
    "true", "0_real", "real_images", "real_image", "natural", "non_ai", "not_ai"
}

FAKE_ALIASES = {
    "fake", "fakes", "deepfake", "deepfakes", "synthetic", "generated",
    "ai", "ai_generated", "aigenerated", "forged", "manipulated",
    "1_fake", "fake_images", "fake_image"
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_name(name: str) -> str:
    name = name.strip().lower()
    for ch in [" ", "-", ".", "(", ")", "[", "]"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def compact_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def infer_label_from_parts(parts: Sequence[str]) -> Optional[int]:
    for part in reversed(parts):
        norm = normalize_name(part)
        comp = compact_name(part)

        if norm in REAL_ALIASES or comp in REAL_ALIASES:
            return 0

        if norm in FAKE_ALIASES or comp in FAKE_ALIASES:
            return 1

    return None


def infer_split_from_parts(parts: Sequence[str]) -> Optional[str]:
    for part in reversed(parts):
        norm = normalize_name(part)
        comp = compact_name(part)

        if norm in SPLIT_ALIASES:
            return SPLIT_ALIASES[norm]

        if comp in SPLIT_ALIASES:
            return SPLIT_ALIASES[comp]

    return None


def scan_dataset_root(root: Path, domain_id: int, domain_name: str) -> List[Dict]:
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = path.relative_to(root).parts[:-1]

        label = infer_label_from_parts(parts)

        if label is None:
            continue

        split = infer_split_from_parts(parts)

        records.append({
            "path": str(path),
            "label": int(label),
            "split": split,
            "domain": int(domain_id),
            "domain_name": domain_name,
        })

    return records


def stratified_split(records, seed=42):
    labels = [r["label"] for r in records]

    train_records, temp_records = train_test_split(
        records,
        train_size=1.0 - cfg.val_ratio - cfg.test_ratio,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    temp_labels = [r["label"] for r in temp_records]

    val_records, test_records = train_test_split(
        temp_records,
        test_size=cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio),
        random_state=seed,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def ensure_splits(records, seed=42):
    by_split = {
        "train": [],
        "val": [],
        "test": [],
        "unknown": [],
    }

    for r in records:
        if r["split"] in ["train", "val", "test"]:
            by_split[r["split"]].append(r)
        else:
            by_split["unknown"].append(r)

    if by_split["train"] and by_split["val"] and by_split["test"]:
        return {
            "train": by_split["train"],
            "val": by_split["val"],
            "test": by_split["test"],
        }

    merged = by_split["train"] + by_split["val"] + by_split["test"] + by_split["unknown"]

    return stratified_split(merged, seed)


def stratified_cap(records, max_samples, seed=42):
    if max_samples is None or len(records) <= max_samples:
        return list(records)

    rng = random.Random(seed)

    real = [r for r in records if r["label"] == 0]
    fake = [r for r in records if r["label"] == 1]

    rng.shuffle(real)
    rng.shuffle(fake)

    half = max_samples // 2
    selected = real[:half] + fake[:max_samples - half]

    if len(selected) < max_samples:
        selected_paths = {r["path"] for r in selected}
        remaining = [r for r in records if r["path"] not in selected_paths]
        rng.shuffle(remaining)
        selected += remaining[:max_samples - len(selected)]

    rng.shuffle(selected)

    return selected


def assert_no_overlap(groups):
    seen = {}

    for name, records in groups.items():
        for r in records:
            p = str(Path(r["path"]).resolve())

            if p in seen:
                raise RuntimeError(f"Data leakage: {p} in {seen[p]} and {name}")

            seen[p] = name


def dataset_summary_rows(dataset_name, role, splits):
    rows = []

    for split, records in splits.items():
        labels = np.array([r["label"] for r in records])

        rows.append({
            "Dataset": dataset_name,
            "Role": role,
            "Split": split,
            "Total": len(records),
            "Real": int(np.sum(labels == 0)) if len(labels) else 0,
            "Fake": int(np.sum(labels == 1)) if len(labels) else 0,
        })

    return rows


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


class RandomJPEGCompression:
    def __init__(self, qmin=15, qmax=90, p=0.75):
        self.qmin = qmin
        self.qmax = qmax
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        q = random.randint(self.qmin, self.qmax)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)

        return Image.open(buffer).convert("RGB")


class RandomGamma:
    def __init__(self, gmin=0.75, gmax=1.35, p=0.35):
        self.gmin = gmin
        self.gmax = gmax
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        gamma = random.uniform(self.gmin, self.gmax)

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.power(np.clip(arr, 0, 1), gamma)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")


class RandomResizeArtifact:
    def __init__(self, scale_min=0.55, scale_max=0.95, p=0.35):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.scale_min, self.scale_max)

        small = img.resize(
            (max(8, int(w * scale)), max(8, int(h * scale))),
            Image.BILINEAR,
        )

        return small.resize((w, h), Image.BILINEAR).convert("RGB")


class RandomGaussianNoiseTensor:
    def __init__(self, sigma_min=0.002, sigma_max=0.035, p=0.30):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        sigma = random.uniform(self.sigma_min, self.sigma_max)

        return torch.clamp(
            x + torch.randn_like(x) * sigma,
            0.0,
            1.0,
        )


def build_train_transform(image_size):
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.88, 1.12)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=8),
        RandomResizeArtifact(p=0.35),
        RandomJPEGCompression(15, 90, p=0.80),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.30),
        T.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.035),
        RandomGamma(p=0.35),
        T.RandomGrayscale(p=0.08),
        T.ToTensor(),
        RandomGaussianNoiseTensor(p=0.30),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (cfg.image_size, cfg.image_size), color=(127, 127, 127))

        if self.transform:
            img = self.transform(img)

        return img, int(rec["label"]), int(rec["domain"]), rec["path"]


class DeepfakeDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_records = None
        self.val_records = None
        self.test_records = None
        self.splits_a = None
        self.splits_b = None
        self.splits_c = None

    def setup(self, stage=None):
        output_tables = Path(cfg.output_dir) / "tables"
        output_tables.mkdir(parents=True, exist_ok=True)

        dataset_a_root = Path(kagglehub.dataset_download(cfg.dataset_a_slug))
        dataset_b_root = Path(kagglehub.dataset_download(cfg.dataset_b_slug))
        dataset_c_root = Path(kagglehub.dataset_download(cfg.dataset_c_slug))

        records_a = scan_dataset_root(dataset_a_root, domain_id=0, domain_name="Dataset_A")
        records_b = scan_dataset_root(dataset_b_root, domain_id=1, domain_name="Dataset_B")
        records_c = scan_dataset_root(dataset_c_root, domain_id=2, domain_name="Dataset_C_Unseen")

        if len(records_a) == 0 or len(records_b) == 0 or len(records_c) == 0:
            raise RuntimeError("One or more datasets returned 0 labeled images. Check folder names.")

        self.splits_a = ensure_splits(records_a, seed=cfg.seed)
        self.splits_b = ensure_splits(records_b, seed=cfg.seed + 1)
        self.splits_c = ensure_splits(records_c, seed=cfg.seed + 2)

        table_rows = []
        table_rows += dataset_summary_rows(
            "Dataset A: manjilkarki/deepfake-and-real-images",
            "Training source domain",
            self.splits_a,
        )
        table_rows += dataset_summary_rows(
            "Dataset B: xhlulu/140k-real-and-fake-faces",
            "Training source domain",
            self.splits_b,
        )
        table_rows += dataset_summary_rows(
            "Dataset C: shivamardeshna image forensics",
            "Unseen test domain",
            self.splits_c,
        )

        pd.DataFrame(table_rows).to_csv(
            output_tables / "dataset_description_and_roles.csv",
            index=False,
        )

        a_train = stratified_cap(self.splits_a["train"], cfg.max_a_train_samples, cfg.seed)
        b_train = stratified_cap(self.splits_b["train"], cfg.max_b_train_samples, cfg.seed + 1)
        a_val = stratified_cap(self.splits_a["val"], cfg.max_val_samples // 2, cfg.seed + 2)
        b_val = stratified_cap(self.splits_b["val"], cfg.max_val_samples // 2, cfg.seed + 3)
        c_test = stratified_cap(self.splits_c["test"], cfg.max_c_test_samples, cfg.seed + 4)

        self.train_records = a_train + b_train
        self.val_records = a_val + b_val
        self.test_records = c_test

        assert_no_overlap({
            "train_A_B": self.train_records,
            "val_A_B": self.val_records,
            "test_C_unseen": self.test_records,
        })

        split_table = pd.DataFrame([
            {
                "Split": "Train",
                "Sources": "Dataset A + Dataset B",
                "Total": len(self.train_records),
                "Real": sum(r["label"] == 0 for r in self.train_records),
                "Fake": sum(r["label"] == 1 for r in self.train_records),
            },
            {
                "Split": "Validation",
                "Sources": "Dataset A + Dataset B",
                "Total": len(self.val_records),
                "Real": sum(r["label"] == 0 for r in self.val_records),
                "Fake": sum(r["label"] == 1 for r in self.val_records),
            },
            {
                "Split": "Unseen Test",
                "Sources": "Dataset C only",
                "Total": len(self.test_records),
                "Real": sum(r["label"] == 0 for r in self.test_records),
                "Fake": sum(r["label"] == 1 for r in self.test_records),
            },
        ])

        split_table.to_csv(
            output_tables / "final_experimental_split.csv",
            index=False,
        )

        records_to_df(self.train_records, "train_A_plus_B").to_csv(
            output_tables / "train_A_plus_B_split.csv",
            index=False,
        )
        records_to_df(self.val_records, "validation_A_plus_B").to_csv(
            output_tables / "validation_A_plus_B_split.csv",
            index=False,
        )
        records_to_df(self.test_records, "unseen_test_C").to_csv(
            output_tables / "unseen_test_C_split.csv",
            index=False,
        )

        print("Train:", len(self.train_records))
        print("Val:", len(self.val_records))
        print("Unseen Test:", len(self.test_records))

    def train_dataloader(self):
        return DataLoader(
            DeepfakeDataset(self.train_records, build_train_transform(cfg.image_size)),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            DeepfakeDataset(self.val_records, build_eval_transform(cfg.image_size)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            DeepfakeDataset(self.test_records, build_eval_transform(cfg.image_size)),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
