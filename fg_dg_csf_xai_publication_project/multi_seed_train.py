import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning as L

from config import cfg
from data import DeepfakeDataModule
from lightning_module import DetectorLightningModule, compute_numpy_metrics
from train import run_one_experiment, run_tta, find_best_or_last_checkpoint


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "eer",
    "tpr_at_1pct_fpr",
    "tpr_at_0_1pct_fpr",
    "ece",
    "brier",
]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def t_critical_975(df):
    # For 3 seeds, df=2, t critical = 4.303.
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            20: 2.086,
            30: 2.042,
        }
        return table.get(df, 1.96)


def metric_mean_std_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))

    if n <= 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "ci95_half_width": 0.0,
            "n": n,
        }

    std = float(np.std(values, ddof=1))
    half = float(t_critical_975(n - 1) * std / math.sqrt(n))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
        "n": n,
    }


def bootstrap_ci_from_predictions(y_true, y_prob, n_boot=1000, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    point = compute_numpy_metrics(y_true, y_prob, cfg.threshold)
    boot_values = {k: [] for k in METRIC_KEYS}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        m = compute_numpy_metrics(y_b, p_b, cfg.threshold)
        for k in METRIC_KEYS:
            boot_values[k].append(float(m[k]))

    rows = []
    for k in METRIC_KEYS:
        vals = np.asarray(boot_values[k], dtype=float)
        if len(vals) == 0:
            rows.append({
                "metric": k,
                "estimate": point.get(k, np.nan),
                "bootstrap_ci95_low": np.nan,
                "bootstrap_ci95_high": np.nan,
                "bootstrap_n": 0,
            })
        else:
            rows.append({
                "metric": k,
                "estimate": float(point[k]),
                "bootstrap_ci95_low": float(np.percentile(vals, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(vals, 97.5)),
                "bootstrap_n": int(len(vals)),
            })
    return pd.DataFrame(rows)


def load_seed_final_model():
    ckpt_path = find_best_or_last_checkpoint("final")
    if ckpt_path is None:
        return None
    lit_model = DetectorLightningModule.load_from_checkpoint(str(ckpt_path), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model.to(device)
    lit_model.eval()
    return lit_model


def train_one_seed(seed, epochs, fixed_split=True, split_seed=42, run_tta_flag=True, force=False):
    print("\n" + "=" * 80)
    print(f"START SEED {seed}")
    print("=" * 80)

    original_seed = cfg.seed
    original_output_dir = cfg.output_dir
    original_checkpoint_dir = cfg.checkpoint_dir
    original_epochs = cfg.epochs

    cfg.output_dir = f"outputs/multi_seed/seed_{seed}"
    cfg.checkpoint_dir = f"checkpoints/multi_seed/seed_{seed}"
    cfg.epochs = epochs

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    metrics_path = Path(cfg.output_dir) / "runs" / "final" / "test_metrics.json"
    tta_path = Path(cfg.output_dir) / "runs" / "final" / "tta_metrics.json"

    try:
        if metrics_path.exists() and not force:
            print(f"Seed {seed}: metrics already exist. Skipping training.")
            if run_tta_flag and not tta_path.exists():
                print("TTA missing. Running TTA only.")
                cfg.seed = split_seed if fixed_split else seed
                dm = DeepfakeDataModule()
                dm.setup()
                cfg.seed = seed
                L.seed_everything(seed, workers=True)
                lit_model = load_seed_final_model()
                if lit_model is not None:
                    run_tta(lit_model, dm, "final")
            return

        if fixed_split:
            cfg.seed = split_seed
            dm = DeepfakeDataModule()
            dm.setup()
            cfg.seed = seed
            L.seed_everything(seed, workers=True)
        else:
            cfg.seed = seed
            L.seed_everything(seed, workers=True)
            dm = DeepfakeDataModule()
            dm.setup()

        lit_model, _best = run_one_experiment(
            "final",
            dm,
            max_epochs=epochs,
            force=force,
        )

        if run_tta_flag:
            run_tta(lit_model, dm, "final")

    finally:
        cfg.seed = original_seed
        cfg.output_dir = original_output_dir
        cfg.checkpoint_dir = original_checkpoint_dir
        cfg.epochs = original_epochs


def collect_metrics(seeds):
    rows = []
    for seed in seeds:
        run_dir = Path(f"outputs/multi_seed/seed_{seed}") / "runs" / "final"
        for setting, filename in [("standard", "test_metrics.json"), ("tta", "tta_metrics.json")]:
            m = load_json(run_dir / filename)
            if m is None:
                continue
            row = {"seed": seed, "setting": setting}
            for k in METRIC_KEYS:
                row[k] = m.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(metrics_df):
    rows = []
    pretty = []
    for setting in sorted(metrics_df["setting"].unique()):
        sub = metrics_df[metrics_df["setting"] == setting]
        for metric in METRIC_KEYS:
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            s = metric_mean_std_ci(vals)
            rows.append({"setting": setting, "metric": metric, **s})
            pretty.append({
                "setting": setting,
                "metric": metric,
                "mean ± std": f"{s['mean']:.4f} ± {s['std']:.4f}",
                "95% CI": f"[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]",
                "n_seeds": s["n"],
            })
    return pd.DataFrame(rows), pd.DataFrame(pretty)


def run_bootstrap_for_seeds(seeds, output_dir, n_boot):
    output_dir = Path(output_dir)
    boot_dir = output_dir / "bootstrap_ci"
    boot_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if n_boot <= 0:
        return pd.DataFrame()

    for seed in seeds:
        run_dir = Path(f"outputs/multi_seed/seed_{seed}") / "runs" / "final"
        for setting, filename in [("standard", "test_predictions.csv"), ("tta", "tta_predictions.csv")]:
            pred_path = run_dir / filename
            if not pred_path.exists():
                continue
            pred = pd.read_csv(pred_path)
            ci = bootstrap_ci_from_predictions(
                pred["label"].values,
                pred["prob_fake"].values,
                n_boot=n_boot,
                seed=seed,
            )
            ci.insert(0, "seed", seed)
            ci.insert(1, "setting", setting)
            ci.to_csv(boot_dir / f"bootstrap_ci_seed_{seed}_{setting}.csv", index=False)
            all_rows.append(ci)

    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(output_dir / "bootstrap_ci_all_seeds.csv", index=False)
    return df


def make_ensemble_predictions(seeds, setting="standard"):
    filename = "test_predictions.csv" if setting == "standard" else "tta_predictions.csv"
    merged = None

    for seed in seeds:
        pred_path = Path(f"outputs/multi_seed/seed_{seed}") / "runs" / "final" / filename
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)[["path", "label", "prob_fake"]].copy()
        df = df.rename(columns={"prob_fake": f"prob_fake_seed_{seed}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["path", "label"], how="inner")

    if merged is None:
        return None
    prob_cols = [c for c in merged.columns if c.startswith("prob_fake_seed_")]
    if not prob_cols:
        return None
    merged["prob_fake_ensemble_mean"] = merged[prob_cols].mean(axis=1)
    return merged


def save_ensemble_metrics(seeds, output_dir, bootstrap_n=1000):
    output_dir = Path(output_dir)
    ensemble_dir = output_dir / "ensemble_predictions"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for setting in ["standard", "tta"]:
        ens = make_ensemble_predictions(seeds, setting)
        if ens is None or len(ens) == 0:
            continue
        ens.to_csv(ensemble_dir / f"ensemble_{setting}_predictions.csv", index=False)
        m = compute_numpy_metrics(
            ens["label"].values,
            ens["prob_fake_ensemble_mean"].values,
            cfg.threshold,
        )
        m["setting"] = f"ensemble_{setting}"
        m["n_images"] = int(len(ens))
        rows.append(m)

        if bootstrap_n > 0:
            ci = bootstrap_ci_from_predictions(
                ens["label"].values,
                ens["prob_fake_ensemble_mean"].values,
                n_boot=bootstrap_n,
                seed=999,
            )
            ci.insert(0, "setting", f"ensemble_{setting}")
            ci.to_csv(output_dir / f"ensemble_{setting}_bootstrap_ci.csv", index=False)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "ensemble_metrics.csv", index=False)
        return df
    return pd.DataFrame()


def plot_mean_ci(summary_df, output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for setting in summary_df["setting"].unique():
        sub = summary_df[summary_df["setting"] == setting].copy()
        x = np.arange(len(sub))
        means = sub["mean"].values
        yerr = np.vstack([means - sub["ci95_low"].values, sub["ci95_high"].values - means])
        plt.figure(figsize=(12, 5))
        plt.bar(x, means, yerr=yerr, capsize=5)
        plt.xticks(x, sub["metric"].values, rotation=30, ha="right")
        plt.ylabel("Metric value")
        plt.title(f"Multi-seed final test metrics: {setting}\nMean with 95% confidence interval")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"multi_seed_mean_ci_{setting}.png", dpi=300)
        plt.close()


def plot_seed_lines(metrics_df, output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    line_metrics = ["accuracy", "f1", "auc", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr", "ece"]

    for setting in metrics_df["setting"].unique():
        sub = metrics_df[metrics_df["setting"] == setting].copy()
        plt.figure(figsize=(10, 5))
        for metric in line_metrics:
            if metric in sub.columns:
                plt.plot(sub["seed"], sub[metric], marker="o", label=metric)
        plt.xlabel("Seed")
        plt.ylabel("Metric value")
        plt.title(f"Seed-wise metric stability: {setting}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"seed_wise_metric_stability_{setting}.png", dpi=300)
        plt.close()


def create_zip(output_dir):
    output_dir = Path(output_dir)
    zip_path = Path("multi_seed_final_results.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", root_dir=str(output_dir))
    print("Created ZIP:", zip_path)
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--fixed_split", action="store_true")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path("multi_seed_final_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "multi_seed_run_config.json")

    for seed in args.seeds:
        train_one_seed(
            seed=seed,
            epochs=args.epochs,
            fixed_split=args.fixed_split,
            split_seed=args.split_seed,
            run_tta_flag=args.tta,
            force=args.force,
        )

    metrics_df = collect_metrics(args.seeds)
    if len(metrics_df) == 0:
        raise RuntimeError("No seed metrics found. Check training logs.")
    metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)

    summary_df, pretty_df = summarize_across_seeds(metrics_df)
    summary_df.to_csv(output_dir / "mean_std_ci95_by_metric.csv", index=False)
    pretty_df.to_csv(output_dir / "mean_std_ci95_pretty_table.csv", index=False)
    pretty_df.to_latex(output_dir / "mean_std_ci95_pretty_table.tex", index=False)

    bootstrap_df = run_bootstrap_for_seeds(args.seeds, output_dir, args.bootstrap_n)
    ensemble_df = save_ensemble_metrics(args.seeds, output_dir, bootstrap_n=args.bootstrap_n)

    plot_mean_ci(summary_df, output_dir)
    plot_seed_lines(metrics_df, output_dir)

    save_json(
        {
            "metrics_by_seed": "metrics_by_seed.csv",
            "mean_std_ci95": "mean_std_ci95_by_metric.csv",
            "pretty_table": "mean_std_ci95_pretty_table.csv",
            "bootstrap_ci": "bootstrap_ci_all_seeds.csv" if len(bootstrap_df) else None,
            "ensemble_metrics": "ensemble_metrics.csv" if len(ensemble_df) else None,
        },
        output_dir / "multi_seed_manifest.json",
    )

    create_zip(output_dir)
    print("\nMulti-seed table:")
    print(pretty_df)


if __name__ == "__main__":
    main()
