"""CLI for extracting GSE211692 discovery matrices, clinical stages, and downloading external GEO cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from highway_bilstm.data.external import download_external_cohorts, load_mimat_mapping
from highway_bilstm.data.gse211692 import (
    download_discovery_cohort,
    extract_32_mirna_dataset,
    extract_full_mirnome,
    parse_clinical_metadata_from_series_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare discovery and external GEO datasets.")
    parser.add_argument("--raw-dir", default="data/GSE211692_RAW", help="Directory containing raw GSE211692 .txt.gz files.")
    parser.add_argument("--mapping-file", default="data/mapping_file.txt", help="Path to GSE211692 mapping_file.txt.")
    parser.add_argument("--meta-train", default="data/meta_train_full.csv", help="Path to meta_train_full.csv.")
    parser.add_argument("--meta-benign", default="data/metadata_full_benign_marked.csv", help="Path to metadata_full_benign_marked.csv.")
    parser.add_argument("--series-matrix", default="data_cache/GSE211692_series_matrix.txt.gz", help="Path to GSE211692_series_matrix.txt.gz.")
    parser.add_argument("--cache-dir", default="data_cache", help="Output cache directory.")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads for extraction.")
    parser.add_argument("--extract-full", action="store_true", help="Extract full 2,550-miRNA NPZ matrix.")
    parser.add_argument("--extract-32", action="store_true", help="Extract 32-miRNA Parquet dataset.")
    parser.add_argument("--parse-clinical", action="store_true", help="Parse clinical TNM stage CSV from GSE211692 series matrix.")
    parser.add_argument("--download-discovery", action="store_true", help="Download GSE211692 series matrix from NCBI GEO.")
    parser.add_argument("--include-raw-tar", action="store_true", help="Also download GSE211692_RAW.tar when downloading discovery cohort.")
    parser.add_argument("--download-external", action="store_true", help="Download the 4 external GPL21263 GEO cohorts.")
    parser.add_argument("--download-all-geo", action="store_true", help="Download all 5 GPL21263 GEO cohorts (GSE211692 + 4 external).")
    parser.add_argument("--export-mimat-map", action="store_true", help="Export MIMAT -> hsa-miR-* mapping JSON.")
    parser.add_argument("--mimat-out", default="configs/gpl21263_mimat_to_mirna.json", help="Output JSON path for MIMAT mapping.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.export_mimat_map:
        mapping = load_mimat_mapping(raw_dir=args.raw_dir)
        out_path = Path(args.mimat_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(mapping.items())), f, indent=2)
        print(f"Exported {len(mapping)} MIMAT mappings to {out_path}")

    if args.extract_full:
        out_npz = extract_full_mirnome(
            args.raw_dir,
            args.mapping_file,
            args.meta_train,
            cache_dir / "full_mirnome_matrix.npz",
            max_workers=args.workers,
        )
        print(f"Saved full miRNome NPZ to {out_npz}")

    if args.extract_32:
        out_pq = extract_32_mirna_dataset(
            args.raw_dir,
            args.mapping_file,
            args.meta_train,
            args.meta_benign,
            cache_dir / "mirna_32_dataset.parquet",
            max_workers=args.workers,
        )
        print(f"Saved 32-miRNA Parquet to {out_pq}")

    if args.parse_clinical:
        out_csv = cache_dir / "GSE211692_clinical_metadata.csv"
        df = parse_clinical_metadata_from_series_matrix(args.series_matrix, output_csv=out_csv)
        print(f"Parsed {len(df)} clinical metadata records to {out_csv}")

    if args.download_discovery or args.download_all_geo:
        disc = download_discovery_cohort(cache_dir, download_raw_tar=args.include_raw_tar)
        for key, path in disc.items():
            print(f"[GSE211692:{key}] Ready at {path}")

    if args.download_external or args.download_all_geo:
        downloaded = download_external_cohorts(cache_dir / "external_cohorts")
        for acc, path in downloaded.items():
            print(f"[{acc}] Ready at {path}")


if __name__ == "__main__":
    main()
