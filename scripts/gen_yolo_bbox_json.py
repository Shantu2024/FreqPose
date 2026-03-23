#!/usr/bin/env python3
"""Build COCO-style detector bbox json from a YOLO model for one ann file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO


def resolve_image_path(data_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    cands = [data_root / p, data_root / "ExLPose" / p]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve image path for {file_name}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("EXLPOSE_DATA_ROOT", str(repo_root / "data" / "ExLPose")),
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    ann_path = Path(args.ann_file)
    data_root = Path(args.data_root)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    images = ann["images"]
    image_paths: List[str] = []
    id_by_path: Dict[str, int] = {}
    for img in images:
        img_path = resolve_image_path(data_root, img["file_name"])
        img_key = str(img_path)
        image_paths.append(img_key)
        id_by_path[img_key] = int(img["id"])

    model = YOLO(args.weights)
    dets: List[dict] = []

    total = len(image_paths)
    for i in range(0, total, args.batch):
        batch_paths = image_paths[i : i + args.batch]
        results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        for r in results:
            img_key = str(Path(r.path))
            image_id = id_by_path[img_key]
            if r.boxes is None:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes_xyxy, confs, clss):
                if c != 0:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                dets.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, w, h],
                        "score": float(s),
                    }
                )
        print(f"[{min(i + args.batch, total)}/{total}] processed")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dets, f)
    print(f"saved: {out_path} ({len(dets)} detections)")


if __name__ == "__main__":
    main()
