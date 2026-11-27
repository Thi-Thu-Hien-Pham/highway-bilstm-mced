"""GSE211692 discovery dataset extraction and clinical metadata parsing."""

from __future__ import annotations

import gzip
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from highway_bilstm.constants import TARGET_32_MIRNAS

GSE211692_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE211nnn/GSE211692/matrix/"
    "GSE211692_series_matrix.txt.gz"
)
GSE211692_RAW_TAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE211nnn/GSE211692/suppl/"
    "GSE211692_RAW.tar"
)


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
    return dest


def download_discovery_cohort(
    output_dir: str | Path,
    download_raw_tar: bool = False,
) -> dict[str, Path]:
    """Download GSE211692 series matrix (and optionally raw TAR archive) from NCBI GEO."""
    out_dir = Path(output_dir)
    downloaded: dict[str, Path] = {
        "series_matrix": _download_file(
            GSE211692_SERIES_MATRIX_URL,
            out_dir / "GSE211692_series_matrix.txt.gz",
        )
    }
    if download_raw_tar:
        downloaded["raw_tar"] = _download_file(
            GSE211692_RAW_TAR_URL,
            out_dir / "GSE211692_RAW.tar",
        )
    return downloaded


def _load_filename_to_organ(mapping_file: str | Path) -> dict[str, str]:
    filename_to_organ: dict[str, str] = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            filename_to_organ[parts[-1]] = parts[1]
    return filename_to_organ


def _get_mirna_names(raw_dir: str | Path) -> list[str]:
    files = sorted(os.listdir(raw_dir))
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")
    sample_file = os.path.join(raw_dir, files[0])
    names: list[str] = []
    with gzip.open(sample_file, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) > 6 and parts[4].startswith("hsa-"):
                names.append(parts[4].strip())
    return names


def extract_full_mirnome(
    raw_dir: str | Path,
    mapping_file: str | Path,
    meta_train_csv: str | Path,
    output_npz: str | Path,
    max_workers: int = 16,
) -> Path:
    """Extract all 2,550 hsa-* miRNA signals from raw GSE211692 files into an NPZ archive."""
    raw_dir = Path(raw_dir)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    mirna_names = _get_mirna_names(raw_dir)
    filename_to_organ = _load_filename_to_organ(mapping_file)
    df_meta = pd.read_csv(meta_train_csv)
    meta_dict = dict(zip(df_meta["samples"], df_meta["class"]))

    raw_files = sorted(os.listdir(raw_dir))

    def _parse(fname: str):
        vals = []
        with gzip.open(raw_dir / fname, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split("\t")
                if len(p) > 6 and p[4].startswith("hsa-"):
                    vals.append(float(p[6]))
        gsm_id = fname.split("_")[0]
        return gsm_id, fname, np.array(vals, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parse, raw_files))

    sample_ids, organs, binary_classes = [], [], []
    matrix = np.zeros((len(results), len(mirna_names)), dtype=np.float32)
    for i, (gsm_id, fname, vals) in enumerate(results):
        sample_ids.append(gsm_id)
        orig_name = fname.split("_", 1)[1].replace(".gz", "")
        organs.append(filename_to_organ.get(orig_name, "Unknown"))
        binary_classes.append(meta_dict.get(gsm_id, "Unknown"))
        matrix[i, :] = vals

    np.savez_compressed(
        output_npz,
        matrix=matrix,
        sample_ids=np.array(sample_ids),
        organs=np.array(organs),
        binary_classes=np.array(binary_classes),
        mirna_names=np.array(mirna_names),
    )
    return output_npz


def extract_32_mirna_dataset(
    raw_dir: str | Path,
    mapping_file: str | Path,
    meta_train_csv: str | Path,
    meta_benign_csv: str | Path,
    output_parquet: str | Path,
    max_workers: int = 16,
) -> Path:
    """Extract the 32 target miRNAs used by 24-rule and miRBiT baselines into Parquet."""
    raw_dir = Path(raw_dir)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    target_set = set(TARGET_32_MIRNAS)
    filename_to_organ = _load_filename_to_organ(mapping_file)
    df_meta = pd.read_csv(meta_train_csv)
    meta_dict = dict(zip(df_meta["samples"], df_meta["class"]))
    df_benign = pd.read_csv(meta_benign_csv)
    benign_dict = dict(zip(df_benign["samples"], df_benign["class"]))

    raw_files = sorted(os.listdir(raw_dir))

    def _parse(fname: str):
        vals = {}
        with gzip.open(raw_dir / fname, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) > 6 and parts[4] in target_set:
                    try:
                        vals[parts[4]] = float(parts[6])
                    except (ValueError, IndexError):
                        pass
        return fname.split("_")[0], fname, vals

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parse, raw_files))

    rows = []
    for gsm_id, fname, vals in results:
        orig_name = fname.split("_", 1)[1].replace(".gz", "")
        row = {
            "sample_id": gsm_id,
            "raw_file": fname,
            "organ": filename_to_organ.get(orig_name, "Unknown"),
            "binary_class": meta_dict.get(gsm_id, "Unknown"),
            "three_class": benign_dict.get(gsm_id, "Unknown"),
        }
        for mirna in TARGET_32_MIRNAS:
            row[mirna] = vals.get(mirna, np.nan)
        rows.append(row)

    pd.DataFrame(rows).to_parquet(output_parquet, index=False)
    return output_parquet


def _normalize_stage_token(raw_stage: str) -> str:
    val = raw_stage.strip().upper()
    if val in {"0", "STAGE 0"}:
        return "Stage 0"
    if val in {"1", "I", "IA", "IB", "STAGE I"}:
        return "Stage I"
    if val in {"2", "II", "IIA", "IIB", "IIC", "STAGE II"}:
        return "Stage II"
    if val in {"3", "III", "IIIA", "IIIB", "IIIC", "STAGE III"}:
        return "Stage III"
    if val in {"4", "IV", "IVA", "IVB", "STAGE IV"}:
        return "Stage IV"
    return "Unstaged"


def parse_clinical_metadata_from_series_matrix(
    series_matrix_gz: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Parse sample IDs, titles, and clinical characteristics from GSE211692_series_matrix.txt.gz."""
    sample_ids: list[str] = []
    titles: list[str] = []
    chars: list[list[str]] = []

    with gzip.open(series_matrix_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break

    data: dict[str, list[str]] = {"sample_id": sample_ids, "title": titles}
    for idx, ch_row in enumerate(chars):
        data[f"char_{idx}"] = ch_row
    df = pd.DataFrame(data)
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def parse_clinical_stage_map(clinical_path: str | Path) -> dict[str, str]:
    """Map GSM sample_id -> normalized TNM stage ('Stage 0'..'Stage IV', 'Unstaged')."""
    path = Path(clinical_path)
    if path.suffix == ".gz":
        df = parse_clinical_metadata_from_series_matrix(path)
    else:
        df = pd.read_csv(path)

    char_cols = [c for c in df.columns if c.startswith("char_")]
    stage_map: dict[str, str] = {}
    stage_re = re.compile(r"stage\s*:\s*([^\s,;]+)", re.IGNORECASE)

    for _, row in df.iterrows():
        sid = str(row["sample_id"]).strip()
        stage_val = "Unstaged"
        for col in char_cols:
            cell = str(row[col])
            match = stage_re.search(cell)
            if match:
                stage_val = _normalize_stage_token(match.group(1))
                break
        stage_map[sid] = stage_val
    return stage_map


def load_full_mirnome_npz(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load full 2,550-miRNA discovery matrix and metadata arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["matrix"],
        data["sample_ids"],
        data["organs"],
        data["binary_classes"],
        list(data["mirna_names"]),
    )


def load_32_mirna_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load the cached 32-miRNA dataset parquet file."""
    return pd.read_parquet(parquet_path)
