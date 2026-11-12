#!/usr/bin/env python3
"""Reproduce the compact historical availability asset from original raw CSVs."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep_dataset.historical_masks import build_mask_bundle, DEFAULT_BUNDLE

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_BUNDLE))
    a = p.parse_args()
    print(json.dumps(build_mask_bundle(a.raw_dir, a.out), indent=2))
