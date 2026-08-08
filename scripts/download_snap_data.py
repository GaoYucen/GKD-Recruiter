#!/usr/bin/env python3
"""Download SNAP check-in datasets used by GKD-Recruiter.

Downloads the official SNAP social-network-with-checkins datasets:
  - Gowalla    (loc-gowalla_edges.txt.gz, loc-gowalla_totalCheckins.txt.gz)
  - Brightkite (loc-brightkite_edges.txt.gz, loc-brightkite_totalCheckins.txt.gz)

Reference: Cho, Myers, and Leskovec (KDD 2011)
https://snap.stanford.edu/data/loc-gowalla.html
https://snap.stanford.edu/data/loc-brightkite.html

Usage:
    python scripts/download_snap_data.py --dataset gowalla [--raw-dir data/raw/gowalla]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

SNAP_BASE = "https://snap.stanford.edu/data"

DATASETS = {
    "gowalla": {
        "edge": "loc-gowalla_edges.txt.gz",
        "checkin": "loc-gowalla_totalCheckins.txt.gz",
        "page": "loc-gowalla.html",
    },
    "brightkite": {
        "edge": "loc-brightkite_edges.txt.gz",
        "checkin": "loc-brightkite_totalCheckins.txt.gz",
        "page": "loc-brightkite.html",
    },
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    try:
        urlretrieve(url, dest)
    except Exception as exc:  # pragma: no cover - network/IO errors
        print(f"[error] failed to download {url}: {exc}", file=sys.stderr)
        raise
    print(f"[ok] saved: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SNAP check-in datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
        help="Which SNAP dataset to download.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory where the dataset will be stored (default: data/raw).",
    )
    args = parser.parse_args()

    entry = DATASETS[args.dataset]
    raw_dir = Path(args.raw_dir) / args.dataset
    print(f"Downloading '{args.dataset}' to {raw_dir} ...")
    page = f"{SNAP_BASE}/{entry['page']}"
    print(f"[info] reference: {page}")
    for key in ("edge", "checkin"):
        url = f"{SNAP_BASE}/{entry[key]}"
        download(url, raw_dir / entry[key])
    print(f"Done. Point config raw_dir to '{raw_dir}' and run build_snap_benchmark.py.")


if __name__ == "__main__":
    main()