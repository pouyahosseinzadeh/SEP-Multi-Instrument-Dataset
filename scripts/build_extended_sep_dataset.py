#!/usr/bin/env python3
"""Command-line wrapper for building the extended SEP dataset."""

from sep_dataset.cli import main


if __name__ == "__main__":
    main(["build", *(__import__("sys").argv[1:])])
