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