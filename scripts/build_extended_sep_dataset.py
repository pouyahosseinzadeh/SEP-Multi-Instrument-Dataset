#!/usr/bin/env python3
"""Command-line wrapper for building the extended SEP dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep_dataset.cli import main


if __name__ == "__main__":
    main(["build", *sys.argv[1:]])
