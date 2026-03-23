#!/usr/bin/env python3
"""Create a merged ExLPose training annotation (LL + WL) with remapped IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = os.environ.get("EXLPOSE_DATA_ROOT", str(repo_root / "data" / "ExLPose"))
    ann_root = Path(data_root) / "Annotations"
    ll_path = ann_root / "ExLPose_train_LL.json"
    wl_path = ann_root / "ExLPose_train_WL.json"
    out_path = ann_root / "ExLPose_train_LL_WL_merged.json"

    with ll_path.open("r", encoding="utf-8") as f:
        ll = json.load(f)
    with wl_path.open("r", encoding="utf-8") as f:
        wl = json.load(f)

    if ll["categories"] != wl["categories"]:
        raise ValueError("LL and WL categories differ; cannot merge safely.")

    ll_images = ll["images"]
    ll_anns = ll["annotations"]
    wl_images = wl["images"]
    wl_anns = wl["annotations"]

    max_img_id = max(img["id"] for img in ll_images) if ll_images else -1
    max_ann_id = max(ann["id"] for ann in ll_anns) if ll_anns else -1
    img_offset = max_img_id + 1
    ann_offset = max_ann_id + 1

    merged_images = list(ll_images)
    merged_anns = list(ll_anns)

    for img in wl_images:
        ni = dict(img)
        ni["id"] = int(img["id"]) + img_offset
        merged_images.append(ni)

    for ann in wl_anns:
        na = dict(ann)
        na["id"] = int(ann["id"]) + ann_offset
        na["image_id"] = int(ann["image_id"]) + img_offset
        merged_anns.append(na)

    merged = dict(ll)
    merged["images"] = merged_images
    merged["annotations"] = merged_anns

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f)

    print(
        f"Saved: {out_path}\n"
        f"images={len(merged_images)} anns={len(merged_anns)} "
        f"(LL {len(ll_images)}/{len(ll_anns)} + WL {len(wl_images)}/{len(wl_anns)})"
    )


if __name__ == "__main__":
    main()
