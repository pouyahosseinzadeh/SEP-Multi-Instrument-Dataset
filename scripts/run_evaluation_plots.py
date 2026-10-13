#!/usr/bin/env python3
"""Command-line wrapper for SEP dataset evaluation plots."""

from sep_dataset.cli import main


if __name__ == "__main__":
    main(["plot", *(__import__("sys").argv[1:])])
