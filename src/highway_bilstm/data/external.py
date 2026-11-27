"""External GPL21263 GEO cohort downloading, parsing, and subgroup labeling."""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from highway_bilstm.constants import EXTERNAL_COHORT_SPECS


def download_external_cohorts(output_dir: str | Path) -> dict[str, Path]:
    """Download the four GPL21263 external validation series matrices from NCBI GEO."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for acc, spec in EXTERNAL_COHORT_SPECS.items():
        out_path = out_dir / spec["matrix_file"]
        if out_path.exists() and out_path.stat().st_size > 100_000:
            downloaded[acc] = out_path
            continue
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
        downloaded[acc] = out_path
    return downloaded


def load_mimat_mapping(
    json_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> dict[str, str]:
    """Load MIMAT probe ID -> mature hsa-miR-* mapping."""
    if json_path is not None and Path(json_path).exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if raw_dir is not None and Path(raw_dir).exists():
        files = sorted(os.listdir(raw_dir))
        if files:
            mimat_to_mirna: dict[str, str] = {}
            with gzip.open(Path(raw_dir) / files[0], "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.split("\t")
                    if len(p) > 5 and p[4].startswith("hsa-"):
                        mimat_to_mirna[p[5].strip()] = p[4].strip()
            return mimat_to_mirna

    raise FileNotFoundError("Provide a valid MIMAT mapping JSON path or GSE211692_RAW directory.")


def parse_external_series_matrix(
    gz_path: str | Path,
    meta_csv_path: str | Path,
    mimat_to_mirna: dict[str, str],
    target_mirnas: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Parse an external GEO series_matrix.txt.gz and align with clinical metadata CSV."""
    df_meta = pd.read_csv(meta_csv_path)
    id_col = "Unnamed: 0" if "Unnamed: 0" in df_meta.columns else df_meta.columns[0]
    sample_to_class = dict(zip(df_meta[id_col], df_meta["class"]))
    sample_to_desc = dict(zip(df_meta[id_col], df_meta["description"]))
    target_set = set(target_mirnas) if target_mirnas is not None else None

    sample_ids: list[str] = []
    table_started = False
    mirna_dict: dict[str, list[float]] = {}

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.replace('"', "").strip() for s in line.split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                table_started = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                break
            elif table_started:
                parts = line.split("\t")
                probe_id = parts[0].replace('"', "").strip()
                if probe_id in mimat_to_mirna:
                    mirna_name = mimat_to_mirna[probe_id]
                    if target_set is not None and mirna_name not in target_set:
                        continue
                    try:
                        mirna_dict[mirna_name] = [float(v.replace('"', "").strip()) for v in parts[1:]]
                    except ValueError:
                        pass

    df_expr = pd.DataFrame(mirna_dict, index=sample_ids)
    valid_samples = [s for s in sample_ids if s in sample_to_class]
    df_expr = df_expr.loc[valid_samples]
    y_binary = np.array([1 if sample_to_class[s] == "cancer" else 0 for s in valid_samples], dtype=np.int64)
    descriptions = np.array([str(sample_to_desc.get(s, "")) for s in valid_samples])
    return df_expr, y_binary, descriptions, valid_samples


def classify_external_control_subgroups(
    cohort_id: str,
    y_binary: np.ndarray,
    descriptions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks `(healthy_mask, benign_mask)` for non-cancer controls."""
    is_ctrl = y_binary == 0
    desc_arr = np.asarray(descriptions, dtype=str)

    if cohort_id == "GSE106817":
        healthy = is_ctrl & np.isin(desc_arr, ["non-Cancer B", "non-Cancer C"])
    elif cohort_id == "GSE112264":
        healthy = is_ctrl & (desc_arr == "non-Cancer")
    elif cohort_id in {"GSE113486", "GSE139031"}:
        healthy = is_ctrl & (desc_arr == "Non-cancer control")
    else:
        healthy = is_ctrl & (np.char.find(np.char.lower(desc_arr), "non-cancer") >= 0)

    benign = is_ctrl & (~healthy)
    return healthy, benign
