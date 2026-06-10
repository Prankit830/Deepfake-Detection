import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# VARIANTS USED IN YOUR PROJECT
# ============================================================

BASELINE_AND_ABLATION_VARIANTS = [
    # Frequency-oriented baselines
    "freqnet_like_baseline",
    "f3net_like_baseline",

    # Compact detector
    "mesonet_baseline",

    # Strong CNN / Transformer baselines
    "efficientnet_b4_baseline",
    "convnext_tiny_baseline",
    "vit_b16_baseline",
    "swin_tiny_baseline",
    "xception_baseline",

    # Internal baselines / ablations
    "spatial_baseline",
    "spatial_frequency",
    "spatial_frequency_patch",
    "no_counterfactual",
    "no_domain",

    # Final proposed model
    "final",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path, copied_files: list):
    src = Path(src)
    dst = Path(dst)

    if not src.exists() or not src.is_file():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    copied_files.append({
        "source": str(src),
        "destination": str(dst),
        "file_name": dst.name,
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 4),
    })

    return True


def copy_pattern(src_dir: Path, pattern: str, dst_dir: Path, copied_files: list):
    src_dir = Path(src_dir)

    if not src_dir.exists():
        return 0

    count = 0

    for src in src_dir.rglob(pattern):
        if src.is_file():
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            safe_copy(src, dst, copied_files)
            count += 1

    return count


def load_json(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# COPY MAIN TABLES
# ============================================================

def copy_main_tables(output_dir, package_dir, copied_files):
    table_src = output_dir / "tables"
    table_dst = package_dir / "tables"

    make_dir(table_dst)

    important_tables = [
        "baseline_comparison.csv",
        "baseline_comparison_with_official_methods.csv",
        "official_baseline_comparison.csv",
        "ablation_study.csv",
        "standard_performance.csv",
        "standard_vs_tta_performance.csv",
        "training_configuration.csv",
        "dataset_description_and_roles.csv",
        "final_experimental_split.csv",
        "confusion_matrix_counts.csv",
        "leakage_shortcut_check_table.csv",
    ]

    copied_count = 0

    for table in important_tables:
        if safe_copy(table_src / table, table_dst / table, copied_files):
            copied_count += 1

    # Also copy LaTeX / Excel versions if present
    for pattern in [
        "*baseline*.tex",
        "*ablation*.tex",
        "*baseline*.xlsx",
        "*ablation*.xlsx",
        "*comparison*.xlsx",
    ]:
        copied_count += copy_pattern(
            table_src,
            pattern,
            table_dst,
            copied_files,
        )

    return copied_count


# ============================================================
# COPY MAIN FIGURES
# ============================================================

def copy_main_figures(output_dir, package_dir, copied_files):
    fig_src = output_dir / "figures"
    fig_dst = package_dir / "figures"

    make_dir(fig_dst)

    important_figures = [
        "baseline_comparison_bar_graph.png",
        "ablation_comparison_bar_graph.png",
        "proposed_architecture_diagram.png",
        "roc_curve.png",
        "low_fpr_roc_zoom.png",
        "score_distribution.png",
        "reliability_diagram.png",
        "confusion_matrix.png",
    ]

    copied_count = 0

    for fig in important_figures:
        if safe_copy(fig_src / fig, fig_dst / fig, copied_files):
            copied_count += 1

    # Catch any other baseline / ablation figures
    for pattern in [
        "*baseline*.png",
        "*ablation*.png",
        "*comparison*.png",
        "*external*.png",
    ]:
        copied_count += copy_pattern(
            fig_src,
            pattern,
            fig_dst,
            copied_files,
        )

    return copied_count


# ============================================================
# COPY RUN METRICS FOR EACH MODEL
# ============================================================

def copy_run_metrics(output_dir, package_dir, copied_files, include_predictions=False):
    runs_src = output_dir / "runs"
    runs_dst = package_dir / "model_run_metrics"

    make_dir(runs_dst)

    rows = []
    copied_count = 0

    for variant in BASELINE_AND_ABLATION_VARIANTS:
        run_dir = runs_src / variant

        if not run_dir.exists():
            continue

        dst_variant_dir = runs_dst / variant
        make_dir(dst_variant_dir)

        # Main test metrics
        metrics_path = run_dir / "test_metrics.json"

        if metrics_path.exists():
            safe_copy(
                metrics_path,
                dst_variant_dir / "test_metrics.json",
                copied_files,
            )
            copied_count += 1

            metrics = load_json(metrics_path)

            if metrics:
                row = {
                    "Variant": variant,
                    "Setting": "standard",
                }
                row.update(metrics)
                rows.append(row)

        # TTA metrics if present
        tta_metrics_path = run_dir / "tta_metrics.json"

        if tta_metrics_path.exists():
            safe_copy(
                tta_metrics_path,
                dst_variant_dir / "tta_metrics.json",
                copied_files,
            )
            copied_count += 1

            metrics = load_json(tta_metrics_path)

            if metrics:
                row = {
                    "Variant": variant,
                    "Setting": "tta",
                }
                row.update(metrics)
                rows.append(row)

        # Lightning logs
        logs_dir = run_dir / "logs"

        if logs_dir.exists():
            copied_count += copy_pattern(
                logs_dir,
                "metrics.csv",
                dst_variant_dir / "logs",
                copied_files,
            )

        # Predictions can be large, so optional
        if include_predictions:
            for pred_file in [
                "test_predictions.csv",
                "tta_predictions.csv",
            ]:
                src = run_dir / pred_file

                if src.exists():
                    safe_copy(
                        src,
                        dst_variant_dir / pred_file,
                        copied_files,
                    )
                    copied_count += 1

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(
            package_dir / "tables" / "all_model_metrics_summary.csv",
            index=False,
        )

    return copied_count


# ============================================================
# COPY EXTERNAL BASELINE / OFFICIAL BASELINE FILES
# ============================================================

def copy_official_external_baseline_files(output_dir, package_dir, copied_files):
    copied_count = 0

    official_pred_src = output_dir / "official_baseline_predictions"
    official_pred_dst = package_dir / "official_baseline_predictions"

    if official_pred_src.exists():
        copied_count += copy_pattern(
            official_pred_src,
            "*.csv",
            official_pred_dst,
            copied_files,
        )

    table_src = output_dir / "tables"
    table_dst = package_dir / "tables"

    for file_name in [
        "official_baseline_comparison.csv",
        "baseline_comparison_with_official_methods.csv",
    ]:
        if safe_copy(table_src / file_name, table_dst / file_name, copied_files):
            copied_count += 1

    return copied_count


# ============================================================
# COPY EXTERNAL DATASET EVALUATION RESULTS
# ============================================================

def copy_external_dataset_evaluation(output_dir, package_dir, copied_files):
    copied_count = 0

    external_src = output_dir / "external_dataset_eval"
    external_dst = package_dir / "external_dataset_evaluation"

    if external_src.exists():
        copied_count += copy_pattern(
            external_src,
            "*.csv",
            external_dst,
            copied_files,
        )

        copied_count += copy_pattern(
            external_src,
            "*.json",
            external_dst,
            copied_files,
        )

        copied_count += copy_pattern(
            external_src,
            "*.png",
            external_dst,
            copied_files,
        )

    # Also support the older external evaluation output folder name
    older_external_src = Path("external_model_dataset_evaluation_outputs")
    older_external_dst = package_dir / "external_model_dataset_evaluation_outputs"

    if older_external_src.exists():
        copied_count += copy_pattern(
            older_external_src,
            "*.csv",
            older_external_dst,
            copied_files,
        )

        copied_count += copy_pattern(
            older_external_src,
            "*.json",
            older_external_dst,
            copied_files,
        )

        copied_count += copy_pattern(
            older_external_src,
            "*.png",
            older_external_dst,
            copied_files,
        )

    return copied_count


# ============================================================
# CREATE COMBINED BASELINE + ABLATION TABLE
# ============================================================

def create_combined_tables(package_dir):
    table_dir = package_dir / "tables"

    baseline_path = table_dir / "baseline_comparison.csv"
    baseline_official_path = table_dir / "baseline_comparison_with_official_methods.csv"
    ablation_path = table_dir / "ablation_study.csv"

    combined_rows = []

    # Prefer official-combined baseline table if available
    if baseline_official_path.exists():
        baseline_df = pd.read_csv(baseline_official_path)
        baseline_df.insert(0, "Table_Type", "Baseline + Official External")
        combined_rows.append(baseline_df)

    elif baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_df.insert(0, "Table_Type", "Baseline")
        combined_rows.append(baseline_df)

    if ablation_path.exists():
        ablation_df = pd.read_csv(ablation_path)
        ablation_df.insert(0, "Table_Type", "Ablation")
        combined_rows.append(ablation_df)

    if combined_rows:
        combined = pd.concat(
            combined_rows,
            ignore_index=True,
            sort=False,
        )

        combined.to_csv(
            table_dir / "combined_baseline_ablation_external_summary.csv",
            index=False,
        )

        try:
            combined.to_excel(
                table_dir / "combined_baseline_ablation_external_summary.xlsx",
                index=False,
            )
        except Exception:
            pass


# ============================================================
# README
# ============================================================

def create_readme(package_dir):
    readme = """
Baseline / Ablation / External Baseline Package
===============================================

This ZIP contains a focused package for paper tables and figures related to:

1. Baseline comparison
2. Ablation study
3. External baseline comparison
4. External dataset evaluation, if available
5. Model-wise metrics and logs

Main folders
------------

tables/
    baseline_comparison.csv
    ablation_study.csv
    official_baseline_comparison.csv
    baseline_comparison_with_official_methods.csv
    combined_baseline_ablation_external_summary.csv
    all_model_metrics_summary.csv

figures/
    baseline_comparison_bar_graph.png
    ablation_comparison_bar_graph.png
    other comparison figures if found

model_run_metrics/
    One folder per trained model variant.
    Contains test_metrics.json, tta_metrics.json if available, and logs.

official_baseline_predictions/
    Official prediction CSV files if available.
    Format: path,label,prob_fake

external_dataset_evaluation/
    Metrics and ROC curves for extra datasets such as FaceForensics++ and Celeb-DF if evaluated.

Notes
-----

- Official F3-Net, FreqNet, SBI, and RECCE results should be added from their official repositories or prediction CSV files.
- Frequency-like and F3Net-like baselines in this project are reproducible proxy baselines, not official implementations.
- For Scopus-level reporting, include both internal ablation results and external baseline comparisons.
"""

    (package_dir / "README_BASELINE_ABLATION_EXTERNAL_PACKAGE.txt").write_text(
        readme.strip(),
        encoding="utf-8",
    )


# ============================================================
# MANIFEST
# ============================================================

def create_manifest(package_dir, copied_files):
    manifest_df = pd.DataFrame(copied_files)

    if len(manifest_df) > 0:
        manifest_df.to_csv(
            package_dir / "MANIFEST.csv",
            index=False,
        )

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Focused package for baseline comparison, ablation study, external baselines, and external dataset evaluation.",
        "included_variants": BASELINE_AND_ABLATION_VARIANTS,
        "number_of_files": len(copied_files),
    }

    save_json(
        metadata,
        package_dir / "PACKAGE_METADATA.json",
    )


# ============================================================
# ZIP
# ============================================================

def create_zip(package_dir, output_zip):
    output_zip = Path(output_zip)

    if output_zip.exists():
        output_zip.unlink()

    shutil.make_archive(
        str(output_zip).replace(".zip", ""),
        "zip",
        root_dir=str(package_dir),
    )

    print("Created ZIP:", output_zip)

    return output_zip


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Main project outputs folder.",
    )

    parser.add_argument(
        "--package_dir",
        type=str,
        default="baseline_ablation_external_package",
        help="Temporary package folder.",
    )

    parser.add_argument(
        "--output_zip",
        type=str,
        default="baseline_ablation_external_outputs.zip",
        help="Final ZIP file.",
    )

    parser.add_argument(
        "--include_predictions",
        action="store_true",
        help="Include test_predictions.csv and tta_predictions.csv. This can make ZIP large.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    package_dir = Path(args.package_dir)

    reset_dir(package_dir)

    make_dir(package_dir / "tables")
    make_dir(package_dir / "figures")
    make_dir(package_dir / "model_run_metrics")
    make_dir(package_dir / "external_dataset_evaluation")
    make_dir(package_dir / "official_baseline_predictions")

    copied_files = []

    print("Copying baseline / ablation / external baseline tables...")
    n_tables = copy_main_tables(
        output_dir,
        package_dir,
        copied_files,
    )

    print("Copying baseline / ablation graphs...")
    n_figures = copy_main_figures(
        output_dir,
        package_dir,
        copied_files,
    )

    print("Copying model run metrics...")
    n_runs = copy_run_metrics(
        output_dir,
        package_dir,
        copied_files,
        include_predictions=args.include_predictions,
    )

    print("Copying official / external baseline files...")
    n_official = copy_official_external_baseline_files(
        output_dir,
        package_dir,
        copied_files,
    )

    print("Copying external dataset evaluation files...")
    n_external = copy_external_dataset_evaluation(
        output_dir,
        package_dir,
        copied_files,
    )

    print("Creating combined summary tables...")
    create_combined_tables(
        package_dir,
    )

    create_readme(
        package_dir,
    )

    create_manifest(
        package_dir,
        copied_files,
    )

    zip_path = create_zip(
        package_dir,
        args.output_zip,
    )

    print("\nDone.")
    print("Tables copied:", n_tables)
    print("Figures copied:", n_figures)
    print("Run metrics copied:", n_runs)
    print("Official baseline files copied:", n_official)
    print("External dataset files copied:", n_external)
    print("Final ZIP:", zip_path)


if __name__ == "__main__":
    main()