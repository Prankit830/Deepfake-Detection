from dataclasses import dataclass
from typing import Optional


@dataclass
class CFG:
    project_name: str = "FG-DG-CSF-XAI-Publication"

    # Reproducibility
    seed: int = 42

    # Input
    image_size: int = 224

    # Training
    batch_size: int = 8
    num_workers: int = 4

    # Main requirement:
    # final proposed model uses 12 epochs.
    # paper/baseline/ablation runs use 3 epochs by default.
    final_epochs: int = 12
    paper_epochs: int = 3
    epochs: int = 12

    lr: float = 1e-4
    weight_decay: float = 5e-4
    dropout: float = 0.45
    label_smoothing: float = 0.03

    # Dataset sample limits
    # Set to None if you want full data.
    max_a_train_samples: Optional[int] = 30000
    max_b_train_samples: Optional[int] = 30000
    max_val_samples: Optional[int] = 8000
    max_c_test_samples: Optional[int] = 15000

    # Auto split if split folders do not exist
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Counterfactual training
    cf_probability: float = 0.85
    cf_loss_weight: float = 0.35
    cf_consistency_weight: float = 0.25

    # Hard counterfactual mining
    hard_cf_fraction: float = 0.50
    hardness_alpha: float = 0.45
    hardness_beta: float = 0.35
    hardness_gamma: float = 0.20

    # Domain adversarial training
    domain_loss_weight: float = 0.25
    domain_grl_final_lambda: float = 1.0

    # Evaluation
    threshold: float = 0.50
    calibration_bins: int = 15
    tta_rounds: int = 3

    # Output paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"

    # Datasets
    dataset_a_slug: str = "manjilkarki/deepfake-and-real-images"
    dataset_b_slug: str = "xhlulu/140k-real-and-fake-faces"
    dataset_c_slug: str = "shivamardeshna/real-and-fake-images-dataset-for-image-forensics"


    EXTERNAL_BENCHMARKS = {
    "FaceForensics++": "/teamspace/studios/this_studio/faceforensics_path",
    "Celeb-DF": "/teamspace/studios/this_studio/celebdf_path",
    "DFDC": "/teamspace/studios/this_studio/dfdc_path",
}

cfg = CFG()
# ============================================================
# External Benchmark Evaluation
# ============================================================

