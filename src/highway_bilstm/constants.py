"""Shared constants, biomarker panels, and cohort specifications."""

from __future__ import annotations

import random
import numpy as np
import torch

SEED: int = 42

INVARIANT_CONTROLS: tuple[str, ...] = (
    "hsa-miR-149-3p",
    "hsa-miR-2861",
    "hsa-miR-4463",
)

CANCER_CLASSES: tuple[str, ...] = (
    "BC", "BL", "BT", "CC", "EC", "GC", "GL", "HC", "LK", "OV", "PC", "PR", "SA"
)

CANCER_FULL_NAMES: dict[str, str] = {
    "BC": "Breast Cancer",
    "BL": "Bladder Cancer",
    "BT": "Biliary Tract Cancer",
    "CC": "Colorectal Cancer",
    "EC": "Esophageal Cancer",
    "GC": "Gastric Cancer",
    "GL": "Glioma / Brain Tumor",
    "HC": "Hepatocellular Carcinoma",
    "LK": "Lung Cancer",
    "OV": "Ovarian Cancer",
    "PC": "Pancreatic Cancer",
    "PR": "Prostate Cancer",
    "SA": "Sarcoma",
}

MIRBIT_PAIRS: tuple[tuple[str, str], ...] = (
    ("hsa-miR-4258", "hsa-miR-4730"),
    ("hsa-miR-663a", "hsa-miR-1228-5p"),
    ("hsa-miR-6787-5p", "hsa-miR-6800-5p"),
    ("hsa-miR-1469", "hsa-miR-1268a"),
)

RULE_PAIRS: tuple[tuple[str, str], ...] = (
    # Group 1: 4 Core miRBiT Rules
    ("hsa-miR-4258", "hsa-miR-4730"),
    ("hsa-miR-663a", "hsa-miR-1228-5p"),
    ("hsa-miR-6787-5p", "hsa-miR-6800-5p"),
    ("hsa-miR-1469", "hsa-miR-1268a"),
    # Group 2: Key Biomarker Anchor Pairs
    ("hsa-miR-5100", "hsa-miR-451a"),
    ("hsa-miR-5100", "hsa-miR-1228-5p"),
    ("hsa-miR-5100", "hsa-miR-21-5p"),
    ("hsa-miR-21-5p", "hsa-miR-451a"),
    ("hsa-miR-1233-5p", "hsa-miR-4532"),
    ("hsa-miR-1343-3p", "hsa-miR-6090"),
    ("hsa-miR-6768-5p", "hsa-miR-1268a"),
    # Group 3: Tissue-Specific Diagnostic & Oncomir Rules
    ("hsa-miR-122-5p", "hsa-miR-21-5p"),
    ("hsa-miR-122-5p", "hsa-miR-451a"),
    ("hsa-miR-375", "hsa-miR-21-5p"),
    ("hsa-miR-375", "hsa-miR-141-3p"),
    ("hsa-miR-200c-3p", "hsa-miR-451a"),
    ("hsa-miR-141-3p", "hsa-miR-21-5p"),
    ("hsa-miR-10b-5p", "hsa-miR-451a"),
    ("hsa-miR-182-5p", "hsa-miR-145-5p"),
    ("hsa-miR-143-3p", "hsa-miR-145-5p"),
    ("hsa-miR-205-5p", "hsa-miR-451a"),
    ("hsa-miR-155-5p", "hsa-miR-21-5p"),
    ("hsa-miR-34a-5p", "hsa-miR-21-5p"),
    ("hsa-miR-221-3p", "hsa-miR-222-3p"),
)

TARGET_32_MIRNAS: tuple[str, ...] = (
    "hsa-miR-4258", "hsa-miR-4730", "hsa-miR-663a", "hsa-miR-1228-5p",
    "hsa-miR-6787-5p", "hsa-miR-6800-5p", "hsa-miR-1469", "hsa-miR-1268a",
    "hsa-miR-5100", "hsa-miR-21-5p", "hsa-miR-451a", "hsa-miR-1233-5p",
    "hsa-miR-1343-3p", "hsa-miR-4532", "hsa-miR-6090", "hsa-miR-6768-5p",
    "hsa-miR-122-5p", "hsa-miR-141-3p", "hsa-miR-200c-3p", "hsa-miR-375",
    "hsa-miR-10b-5p", "hsa-miR-182-5p", "hsa-miR-199a-3p", "hsa-miR-143-3p",
    "hsa-miR-145-5p", "hsa-miR-205-5p", "hsa-miR-155-5p", "hsa-miR-34a-5p",
    "hsa-miR-92a-3p", "hsa-miR-196a-5p", "hsa-miR-221-3p", "hsa-miR-222-3p",
)

FIXED_EXTERNAL_RULE_THRESHOLD: float = 9.0 / 24.0

EXTERNAL_COHORT_SPECS: dict[str, dict[str, str]] = {
    "GSE106817": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106817/matrix/GSE106817_series_matrix.txt.gz",
        "matrix_file": "GSE106817_series_matrix.txt.gz",
        "meta_file": "GSE106817_metadatah.csv",
        "description": "Ovarian & Multi-Cancer Cohort (N=4,046)",
        "target_organ": "Ovarian Cancer",
        "target_code": "OV",
    },
    "GSE112264": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112264/matrix/GSE112264_series_matrix.txt.gz",
        "matrix_file": "GSE112264_series_matrix.txt.gz",
        "meta_file": "GSE112264_metadatah.csv",
        "description": "Prostate & Multi-Cancer Male Cohort (N=1,591)",
        "target_organ": "Prostate Cancer",
        "target_code": "PR",
    },
    "GSE113486": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE113nnn/GSE113486/matrix/GSE113486_series_matrix.txt.gz",
        "matrix_file": "GSE113486_series_matrix.txt.gz",
        "meta_file": "GSE113486_metadatah.csv",
        "description": "Bladder & Multi-Cancer Cohort (N=972)",
        "target_organ": "Bladder Cancer",
        "target_code": "BL",
    },
    "GSE139031": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139031/matrix/GSE139031_series_matrix.txt.gz",
        "matrix_file": "GSE139031_series_matrix.txt.gz",
        "meta_file": "GSE139031_metadatah.csv",
        "description": "Glioma/CNS & Benign Control Cohort (N=580)",
        "target_organ": "Glioma",
        "target_code": "GL",
    },
}

EXTERNAL_DESC_TO_CODE: dict[str, str] = {
    "Breast Cancer": "BC",
    "Bladder Cancer": "BL",
    "Biliary Tract Cancer": "BT",
    "Colorectal Cancer": "CC",
    "Esophageal Cancer": "EC",
    "Gastric Cancer": "GC",
    "Glioma": "GL",
    "Hepatocellular Carcinoma": "HC",
    "Lung Cancer": "LK",
    "Ovarian Cancer": "OV",
    "OV_others": "OV",
    "Pancreatic Cancer": "PC",
    "Prostate Cancer": "PR",
    "Sarcoma": "SA",
}


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
