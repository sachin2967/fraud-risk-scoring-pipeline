"""Create a consistent, non-overwriting online backup of the prediction ledger."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.store import DecisionStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("Source ledger does not exist.")
    DecisionStore(args.source).backup(args.output)
    print(f"Consistent backup written to {args.output}")


if __name__ == "__main__":
    main()
