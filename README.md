# FreqPose: Frequency-Guided Enhancement for Deployment-Ready Low-Light Human Pose Estimation

This repository contains the FreqPose codebase built on top of [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose). It includes the Frequency-Guided Enhancement (FGE) module, ExLPose training and evaluation configs, detector-assisted evaluation scripts, and ablation components used for low-light human pose estimation experiments.

## Scope

This project supports:

- full FGE training with HRNet-W32 and ResNet-50
- ExLPose mixed low-light + well-lit training
- GT-box evaluation on ExLPose and ExLPose-OCN
- detector-box evaluation with YOLO-generated person boxes
- ablations for GLIC, LRBD, DCC, and combined variants

## Base Repository

Start from upstream MMPose:

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
```

Then apply or copy the FreqPose files from this repository on top of that MMPose checkout.

## Environment Setup

Create the public conda environment:

```bash
conda env create -f environment.yml
conda activate freqpose
```

Install the repo in editable mode:

```bash
pip install -v -e .
```

## Dataset Setup

Download ExLPose from the official project page:

- <https://cg.postech.ac.kr/research/ExLPose/>

If needed, you may also keep a mirrored backup source for internal use, but the public instructions should follow the official ExLPose release page first.

Place it inside the MMPose repo at:

```text
data/ExLPose/
```

Expected layout:

```text
data/ExLPose/
├── Annotations/
│   ├── ExLPose_train_LL.json
│   ├── ExLPose_train_WL.json
│   ├── ExLPose_test_LL-N.json
│   ├── ExLPose_test_LL-H.json
│   ├── ExLPose_test_LL-E.json
│   ├── ExLPose_test_LL-A.json
│   ├── ExLPose_test_WL.json
│   ├── ExLPose-OC_test_A7M3.json
│   └── ExLPose-OC_test_RICOH3.json
└── ...
```

Set:

```bash
export EXLPOSE_DATA_ROOT=$PWD/data/ExLPose
```

Create the mixed training annotation:

```bash
python scripts/prepare_exlpose_train_mixed.py
```

## Checkpoints

Use a dedicated checkpoint folder inside the MMPose repo:

```text
checkpoints/
```

Recommended filenames:

```text
checkpoints/hrnet_fge_best.pth
checkpoints/res50_fge_best.pth
checkpoints/yolo_exlpose_best.pt
```

You may either:

- download released best checkpoints into `checkpoints/`
- or train new checkpoints and use the outputs from `work_dirs/`

## Training

HRNet-W32 + full FGE:

```bash
bash scripts/train_exlpose_fge_hrnet.sh
```

ResNet-50 + full FGE:

```bash
bash scripts/train_exlpose_fge_res50.sh
```

## Evaluation

GT-box ExLPose split evaluation:

```bash
bash scripts/eval_exlpose_fge_splits.sh checkpoints/hrnet_fge_best.pth hrnet
```

GT-box ExLPose + OCN evaluation:

```bash
bash scripts/eval_exlpose_fge_all.sh checkpoints/hrnet_fge_best.pth hrnet
```

Detector-box evaluation:

```bash
bash scripts/eval_exlpose_fge_all_detector.sh checkpoints/hrnet_fge_best.pth hrnet
```

## Detector Pipeline

Prepare and train the YOLO detector:

```bash
bash scripts/train_yolo_exlpose_det.sh
```

Generate detector-box JSON files:

```bash
python scripts/gen_yolo_bbox_json.py
```

## TensorBoard

Launch TensorBoard for training logs:

```bash
bash scripts/tb_fge.sh
```

## Core FreqPose Files

Model and integration:

- `mmpose/models/necks/fge.py`
- `mmpose/models/pose_estimators/base.py`
- `mmpose/models/pose_estimators/topdown.py`

Dataset wiring:

- `configs/_base_/datasets/exlpose.py`
- `mmpose/datasets/datasets/body/exlpose_dataset.py`

Configs:

- `configs/body_2d_keypoint/topdown_heatmap/exlpose/`

Scripts:

- `scripts/prepare_exlpose_train_mixed.py`
- `scripts/train_exlpose_fge_hrnet.sh`
- `scripts/train_exlpose_fge_res50.sh`
- `scripts/eval_exlpose_fge_splits.sh`
- `scripts/eval_exlpose_fge_all.sh`
- `scripts/eval_exlpose_fge_all_detector.sh`

## Notes

- This repository is a project overlay on top of MMPose, not a replacement for the full upstream framework history.
- `EXLPOSE_DATA_ROOT` is the primary dataset path hook used by the ExLPose configs and scripts.
- `work_dirs/`, `runs/`, local thesis assets, and generated outputs should not be committed to the clean public release.

## Acknowledgements

- [MMPose](https://github.com/open-mmlab/mmpose) for the upstream pose estimation framework this project builds on.
- [ExLPose](https://cg.postech.ac.kr/research/ExLPose/) for the dataset and the low-light pose estimation benchmark that this repository uses for training and evaluation.
