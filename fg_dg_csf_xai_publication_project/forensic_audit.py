import argparse
import hashlib
import json
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ExifTags

ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors

from config import cfg
from data import DeepfakeDataModule


def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_to_int(hash_obj):
    return int(str(hash_obj), 16)


def hamming_int(a, b):
    return int(a ^ b).bit_count()


class BKTree:
    def __init__(self, distance_func):
        self.distance_func = distance_func
        self.tree = None

    def add(self, item):
        if self.tree is None:
            self.tree = (item, {})
            return

        node, children = self.tree

        while True:
            dist = self.distance_func(item[0], node[0])

            if dist in children:
                node, children = children[dist]
            else:
                children[dist] = (item, {})
                break

    def search(self, item, threshold):
        if self.tree is None:
            return []

        results = []
        nodes = [self.tree]

        while nodes:
            node, children = nodes.pop()
            dist = self.distance_func(item[0], node[0])

            if dist <= threshold:
                results.append((node, dist))

            low = dist - threshold
            high = dist + threshold

            for d, child in children.items():
                if low <= d <= high:
                    nodes.append(child)

        return results


def image_inventory(records, split_name, table_dir):
    rows = []

    for r in tqdm(records, desc=f"Inventory {split_name}"):
        path = Path(r["path"])

        try:
            img0 = Image.open(path)
            img = img0.convert("RGB")
            w, h = img.size

            exif = {}

            try:
                exif_raw = img0.getexif()

                for k, v in exif_raw.items():
                    tag = ExifTags.TAGS.get(k, str(k))
                    exif[tag] = str(v)

            except Exception:
                pass

            quant_hash = ""

            try:
                q = img0.quantization

                if q:
                    quant_hash = hashlib.md5(str(q).encode()).hexdigest()

            except Exception:
                pass

            ph = imagehash.phash(img)

            rows.append({
                "split": split_name,
                "path": str(path),
                "label": int(r["label"]),
                "domain_name": r.get("domain_name", ""),
                "extension": path.suffix.lower(),
                "file_size_kb": round(path.stat().st_size / 1024, 3),
                "width": int(w),
                "height": int(h),
                "megapixels": round((w * h) / 1e6, 4),
                "aspect_ratio": round(w / max(1, h), 4),
                "md5": file_md5(path),
                "phash": str(ph),
                "phash_int": hash_to_int(ph),
                "exif_software": exif.get("Software", ""),
                "exif_make": exif.get("Make", ""),
                "exif_model": exif.get("Model", ""),
                "jpeg_quant_hash": quant_hash,
            })

        except Exception as e:
            rows.append({
                "split": split_name,
                "path": str(path),
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        table_dir / f"{split_name}_image_inventory.csv",
        index=False,
    )

    return df


def exact_md5_leakage(train_df, test_df, table_dir):
    train_md5 = set(train_df["md5"].dropna().tolist())

    leaks = test_df[
        test_df["md5"].isin(train_md5)
    ].copy()

    leaks.to_csv(
        table_dir / "exact_md5_train_test_leakage.csv",
        index=False,
    )

    return leaks


def save_pair_grid(pair_df, save_path, max_pairs=8, image_size=160):
    if len(pair_df) == 0:
        return

    n = min(max_pairs, len(pair_df))

    plt.figure(figsize=(8, n * 3))

    for i in range(n):
        row = pair_df.iloc[i]

        img_test = Image.open(row["test_image"]).convert("RGB").resize((image_size, image_size))
        img_train = Image.open(row["train_image"]).convert("RGB").resize((image_size, image_size))

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(img_test)
        plt.title(f"Test\nDist={row['distance']}")
        plt.axis("off")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(img_train)
        plt.title(f"Train\nSim={row['similarity_score']:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def phash_leakage(train_df, test_df, table_dir, fig_dir, threshold=5, max_pairs=50000):
    train_df = train_df.dropna(subset=["phash_int"]).copy()
    test_df = test_df.dropna(subset=["phash_int"]).copy()

    tree = BKTree(hamming_int)

    for row in tqdm(train_df.itertuples(), total=len(train_df), desc="Build train pHash tree"):
        tree.add((int(row.phash_int), row.path))

    pairs = []

    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Search test pHash"):
        matches = tree.search((int(row.phash_int), row.path), threshold)

        for match_item, dist in matches:
            _h, train_path = match_item

            pairs.append({
                "test_image": row.path,
                "train_image": train_path,
                "distance": int(dist),
                "similarity_score": 1.0 - int(dist) / 64.0,
            })

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    pair_df = pd.DataFrame(pairs)

    pair_df.to_csv(
        table_dir / "train_test_phash_near_duplicates.csv",
        index=False,
    )

    if len(pair_df):
        plt.figure(figsize=(8, 5))
        plt.hist(pair_df["distance"], bins=20, alpha=0.75)
        plt.xlabel("pHash Hamming Distance")
        plt.ylabel("Pairs")
        plt.title("Train-Test Near Duplicate pHash Distance")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "train_test_phash_distance_histogram.png", dpi=300)
        plt.close()

        save_pair_grid(
            pair_df,
            fig_dir / "train_test_near_duplicate_examples.png",
        )

    return pair_df


def metadata_resolution_analysis(train_df, test_df, table_dir, fig_dir):
    combined = pd.concat([
        train_df.assign(group="Train"),
        test_df.assign(group="Unseen Test"),
    ], ignore_index=True)

    combined.to_csv(
        table_dir / "compression_resolution_metadata_full.csv",
        index=False,
    )

    for col, fname, title, xlabel in [
        ("megapixels", "resolution_distribution_comparison.png", "Resolution Distribution Comparison", "Megapixels"),
        ("file_size_kb", "file_size_distribution_comparison.png", "File Size Distribution Comparison", "File Size KB"),
        ("aspect_ratio", "aspect_ratio_distribution_comparison.png", "Aspect Ratio Distribution Comparison", "Aspect Ratio"),
    ]:
        plt.figure(figsize=(8, 5))

        for group in ["Train", "Unseen Test"]:
            sub = combined[combined["group"] == group]

            if col in sub.columns:
                plt.hist(sub[col].dropna(), bins=40, alpha=0.5, label=group)

        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=300)
        plt.close()

    ext_table = (
        combined
        .groupby(["group", "extension"])
        .size()
        .reset_index(name="count")
    )

    ext_table.to_csv(
        table_dir / "file_extension_distribution.csv",
        index=False,
    )

    rows = []

    for field in ["exif_software", "exif_make", "exif_model", "jpeg_quant_hash", "extension"]:
        train_vals = set(train_df[field].dropna().astype(str)) if field in train_df.columns else set()
        test_vals = set(test_df[field].dropna().astype(str)) if field in test_df.columns else set()

        train_vals.discard("")
        test_vals.discard("")

        overlap = train_vals.intersection(test_vals)

        rows.append({
            "field": field,
            "unique_train_values": len(train_vals),
            "unique_test_values": len(test_vals),
            "overlap_values": len(overlap),
            "overlap_examples": "; ".join(list(overlap)[:10]),
        })

    metadata_df = pd.DataFrame(rows)

    metadata_df.to_csv(
        table_dir / "metadata_leakage_check.csv",
        index=False,
    )

    return metadata_df


def face_embedding_overlap(train_records, test_records, table_dir, fig_dir, max_train_faces=20000, max_test_faces=15000, threshold=0.65):
    try:
        import torch
        import torch.nn.functional as F
        from facenet_pytorch import MTCNN, InceptionResnetV1

    except Exception as e:
        msg = f"Face embedding skipped. Install with: pip install facenet-pytorch\nError: {e}"
        (table_dir / "face_embedding_identity_overlap_SKIPPED.txt").write_text(msg)
        print(msg)
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        device=device,
    )

    embedder = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(device)

    def embed(records, name, max_n):
        rows = []
        recs = records if max_n is None else records[:max_n]

        for r in tqdm(recs, desc=f"Face embeddings {name}"):
            try:
                img = Image.open(r["path"]).convert("RGB")
                face = mtcnn(img)

                if face is None:
                    continue

                face = face.unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = embedder(face)
                    emb = F.normalize(emb, dim=1)[0].cpu().numpy()

                rows.append({
                    "path": r["path"],
                    "label": int(r["label"]),
                    "embedding": emb,
                })

            except Exception:
                continue

        return rows

    train_rows = embed(train_records, "train", max_train_faces)
    test_rows = embed(test_records, "test", max_test_faces)

    if len(train_rows) == 0 or len(test_rows) == 0:
        return pd.DataFrame()

    train_emb = np.stack([r["embedding"] for r in train_rows])
    test_emb = np.stack([r["embedding"] for r in test_rows])

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(train_emb)
    dist, idx = nn.kneighbors(test_emb)

    rows = []

    for i in range(len(test_rows)):
        sim = 1.0 - float(dist[i][0])
        tr_i = int(idx[i][0])

        if sim >= threshold:
            rows.append({
                "test_image": test_rows[i]["path"],
                "train_image": train_rows[tr_i]["path"],
                "cosine_similarity": sim,
                "test_label": test_rows[i]["label"],
                "train_label": train_rows[tr_i]["label"],
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        table_dir / "face_embedding_identity_overlap_candidates.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(1.0 - dist[:, 0], bins=40, alpha=0.75)
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold}")
    plt.xlabel("Nearest Train Face Cosine Similarity")
    plt.ylabel("Test Images")
    plt.title("Face-Embedding Similarity: Train vs Test")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "face_embedding_similarity_histogram.png", dpi=300)
    plt.close()

    return df


def create_zip(root_dir):
    zip_path = Path("forensic_leakage_audit_outputs.zip")

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        str(zip_path).replace(".zip", ""),
        "zip",
        root_dir=str(root_dir),
    )

    print("ZIP created:", zip_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phash_threshold", type=int, default=5)
    parser.add_argument("--face_similarity_threshold", type=float, default=0.65)
    parser.add_argument("--max_train_faces", type=int, default=20000)
    parser.add_argument("--max_test_faces", type=int, default=15000)
    parser.add_argument("--skip_face", action="store_true")
    args = parser.parse_args()

    output_root = Path(cfg.output_dir) / "forensic_leakage_audit"
    table_dir = output_root / "tables"
    fig_dir = output_root / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dm = DeepfakeDataModule()
    dm.setup()

    train_df = image_inventory(dm.train_records, "train_A_plus_B", table_dir)
    test_df = image_inventory(dm.test_records, "unseen_test_C", table_dir)

    exact_leaks = exact_md5_leakage(train_df, test_df, table_dir)
    phash_pairs = phash_leakage(train_df, test_df, table_dir, fig_dir, threshold=args.phash_threshold)
    metadata_df = metadata_resolution_analysis(train_df, test_df, table_dir, fig_dir)

    if args.skip_face:
        face_df = pd.DataFrame()
    else:
        face_df = face_embedding_overlap(
            dm.train_records,
            dm.test_records,
            table_dir,
            fig_dir,
            max_train_faces=args.max_train_faces,
            max_test_faces=args.max_test_faces,
            threshold=args.face_similarity_threshold,
        )

    summary = {
        "train_images_checked": int(len(train_df)),
        "test_images_checked": int(len(test_df)),
        "exact_md5_leakage_count": int(len(exact_leaks)),
        "phash_near_duplicate_pairs": int(len(phash_pairs)),
        "unique_test_images_phash_similar_to_train": int(phash_pairs["test_image"].nunique()) if len(phash_pairs) else 0,
        "face_identity_overlap_candidates": int(len(face_df)),
        "metadata_fields_checked": int(len(metadata_df)),
    }

    with open(table_dir / "forensic_leakage_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(
        table_dir / "forensic_leakage_audit_summary.csv",
        index=False,
    )

    print("Audit summary:", summary)

    create_zip(output_root)


if __name__ == "__main__":
    main()
