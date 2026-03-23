#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from mmpose.datasets.datasets.utils import parse_pose_metainfo


def parse_args():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.environ.get('EXLPOSE_DATA_ROOT', os.path.join(repo_root, 'data', 'ExLPose'))
    p = argparse.ArgumentParser(
        description='Visualize ExLPose/OCN with GT multi-person boxes in one image.')
    p.add_argument('--ckpt', required=True)
    p.add_argument(
        '--cfg',
        default='configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py',
    )
    p.add_argument(
        '--split',
        default='LL-A',
        choices=['LL-N', 'LL-H', 'LL-E', 'LL-A', 'WL', 'A7M3', 'RICOH3'])
    p.add_argument('--n', type=int, default=-1)
    p.add_argument('--ann-root', default=os.path.join(data_root, 'Annotations'))
    p.add_argument('--img-root', default=data_root)
    p.add_argument('--out-dir', default='release/protocol_gt_vis_multi_fge')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--kpt-thr', type=float, default=0.35)
    p.add_argument('--bbox-thr', type=float, default=0.0)
    p.add_argument('--radius', type=int, default=4)
    p.add_argument('--thickness', type=int, default=2)
    return p.parse_args()


def split_to_ann_file(split: str) -> str:
    if split in {'LL-N', 'LL-H', 'LL-E', 'LL-A'}:
        return f'ExLPose_test_{split}.json'
    if split == 'WL':
        return 'ExLPose_test_WL.json'
    return f'ExLPose-OC_test_{split}.json'


def resolve_image_path(img_root: str, file_name: str) -> str:
    p = file_name
    cands = [
        os.path.join(img_root, p),
        os.path.join(img_root, 'ExLPose', p),
        os.path.join(img_root, 'ExLPose-OCN', p),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f'Image not found. Tried: {cands}')


def main():
    args = parse_args()
    ann_file = os.path.join(args.ann_root, split_to_ann_file(args.split))

    with open(ann_file, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    images = {im['id']: im for im in ann['images']}
    by_img = defaultdict(list)
    for a in ann['annotations']:
        by_img[a['image_id']].append(a)

    all_images = sorted(images.values(), key=lambda x: x['file_name'])
    selected = all_images if args.n <= 0 else all_images[:args.n]

    out_split = os.path.join(args.out_dir, args.split)
    os.makedirs(out_split, exist_ok=True)

    inferencer = MMPoseInferencer(
        pose2d=args.cfg,
        pose2d_weights=args.ckpt,
        det_model='whole_image',
        device=args.device,
    )
    exlpose_meta = parse_pose_metainfo(
        dict(from_file='configs/_base_/datasets/exlpose.py'))
    inferencer.inferencer.visualizer.set_dataset_meta(
        exlpose_meta, skeleton_style='mmpose')

    for im in selected:
        img_path = resolve_image_path(args.img_root, im['file_name'])

        anns = by_img.get(im['id'], [])
        boxes = []
        for a in anns:
            x, y, w, h = a['bbox']
            if w <= 2 or h <= 2:
                continue
            boxes.append([x, y, x + w, y + h])
        if not boxes:
            continue

        bboxes = np.array(boxes, dtype=np.float32)
        result = next(
            inferencer(
                img_path,
                bboxes=[bboxes],
                show=False,
                return_vis=True,
                draw_bbox=True,
                kpt_thr=args.kpt_thr,
                bbox_thr=args.bbox_thr,
                pose_based_nms=False,
                radius=args.radius,
                thickness=args.thickness,
                num_instances=-1,
            ))

        vis = result['visualization'][0]
        if vis.dtype != np.uint8:
            vis = np.clip(vis, 0, 255).astype(np.uint8)
        if vis.ndim == 3 and vis.shape[2] == 3:
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        out_name = os.path.basename(im['file_name'])
        cv2.imwrite(os.path.join(out_split, out_name), vis)
        print(f'[{args.split}] saved {out_name} ({len(boxes)} persons)')

    print(f'Done. Output: {out_split}')


if __name__ == '__main__':
    main()
