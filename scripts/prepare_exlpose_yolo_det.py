#!/usr/bin/env python3
"""Build a YOLO detection dataset from ExLPose COCO-style annotations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_anns_by_image(anns: Iterable[dict]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = defaultdict(list)
    for ann in anns:
        if int(ann.get("category_id", 1)) != 1:
            continue
        out[int(ann["image_id"])].append(ann)
    return out


def _write_yolo_label(label_path: Path, anns: List[dict], w: int, h: int) -> int:
    _ensure_parent(label_path)
    lines: List[str] = []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        xc = (x + bw * 0.5) / max(w, 1)
        yc = (y + bh * 0.5) / max(h, 1)
        nw = bw / max(w, 1)
        nh = bh / max(h, 1)
        lines.append(f"0 {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _build_split(
    split_name: str,
    sources: List[Tuple[str, Path]],
    data_root: Path,
    out_root: Path,
    link_mode: str,
) -> Tuple[int, int]:
    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_boxes = 0

    for tag, ann_path in sources:
        coco = _load_coco(ann_path)
        anns_by_img = _group_anns_by_image(coco["annotations"])

        for img in coco["images"]:
            img_id = int(img["id"])
            src = data_root / "ExLPose" / img["file_name"]
            if not src.exists():
                src = data_root / img["file_name"]
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")

            suffix = Path(img["file_name"]).suffix
            stem = f"{tag}_{img_id:08d}"
            dst_img = out_img_dir / f"{stem}{suffix}"
            dst_lbl = out_lbl_dir / f"{stem}.txt"

            _link_or_copy(src, dst_img, link_mode)
            num_boxes += _write_yolo_label(
                dst_lbl, anns_by_img.get(img_id, []), int(img["width"]), int(img["height"])
            )
            num_images += 1

    return num_images, num_boxes


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = os.environ.get("EXLPOSE_DATA_ROOT", str(repo_root / "data" / "ExLPose"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=default_root)
    parser.add_argument("--ann-root", default=str(Path(default_root) / "Annotations"))
    parser.add_argument("--out-root", default=str(repo_root / "data" / "ExLPoseDetector"))
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_root = Path(args.ann_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_sources = [
        ("ll", ann_root / "ExLPose_train_LL.json"),
        ("wl", ann_root / "ExLPose_train_WL.json"),
    ]
    val_sources = [("lla", ann_root / "ExLPose_test_LL-A.json")]

    for ann_path in [p for _, p in train_sources + val_sources]:
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    tr_imgs, tr_boxes = _build_split("train", train_sources, data_root, out_root, args.link_mode)
    va_imgs, va_boxes = _build_split("val", val_sources, data_root, out_root, args.link_mode)

    yaml_path = out_root / "exlpose_det.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Built YOLO dataset at: {out_root}")
    print(f"train: images={tr_imgs} boxes={tr_boxes}")
    print(f"val:   images={va_imgs} boxes={va_boxes}")
    print(f"yaml:  {yaml_path}")


if __name__ == "__main__":
    main()
