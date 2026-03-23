#!/usr/bin/env python3
"""Create a COCO annotation copy with `area` filled from bbox when missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for ann in data.get("annotations", []):
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = float(max(w, 0.0) * max(h, 0.0))
            changed += 1

    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"saved={dst} changed={changed}")


if __name__ == "__main__":
    main()
