# FreqPose: Frequency-Guided Enhancement for Deployment-Ready Low-Light Human Pose Estimation

This repository contains the FreqPose codebase built on top of [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose). It includes the Frequency-Guided Enhancement (FGE) module, ExLPose training and evaluation configs, detector-assisted evaluation scripts, and ablation components used for low-light human pose estimation experiments.

Released checkpoints: [Google Drive download](https://drive.google.com/file/d/180nyWHlX3snr18YKf71ycfk9-hYzqFQm/view?usp=drive_link)

## Scope

This project supports:

- full FGE training with HRNet-W32 and ResNet-50
- ExLPose mixed low-light + well-lit training
- GT-box evaluation on ExLPose and ExLPose-OCN
- detector-box evaluation with YOLO-generated person boxes
- ablations for GLIC, LRBD, DCC, and combined variants

## Setup

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
git clone https://github.com/Shantu2024/FreqPose.git
conda env create -f FreqPose/environment.yml
conda activate freqpose
pip install -v -e .
```

Run the remaining FreqPose scripts from the MMPose root after this setup.

## Dataset Setup

Download ExLPose from the official project page:

- <https://cg.postech.ac.kr/research/ExLPose/>

If needed, you may also keep a mirrored backup source for internal use, but the public instructions should follow the official ExLPose release page first.

Place it inside the MMPose repo at:

```text
data/ExLPose/
```

Expected dataset layout:

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

Expected working layout after setup:

```text
mmpose/
├── FreqPose/
│   ├── README.md
│   ├── environment.yml
│   ├── configs/
│   ├── mmpose/
│   └── scripts/
├── checkpoints/
│   ├── hrnet_fge_best.pth
│   ├── res50_fge_best.pth
│   └── yolo_exlpose_best.pt
├── data/
│   ├── ExLPose/
│   │   ├── Annotations/
│   │   └── ...
│   └── ExLPoseDetector/
├── tools/
├── work_dirs/
└── ...
```

Notes:

- `checkpoints/` stores released pose checkpoints and the trained detector weights.
- `data/ExLPose/` stores the downloaded ExLPose dataset.
- `data/ExLPoseDetector/` is created by the detector preparation script.
- `work_dirs/` is created automatically by MMPose and YOLO training runs.

Set:

```bash
export EXLPOSE_DATA_ROOT=$PWD/data/ExLPose
```

Create the mixed training annotation:

```bash
python FreqPose/scripts/prepare_exlpose_train_mixed.py
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

- download the released best checkpoints from [Google Drive](https://drive.google.com/file/d/180nyWHlX3snr18YKf71ycfk9-hYzqFQm/view?usp=drive_link) into `checkpoints/`
- or train new checkpoints and use the outputs from `work_dirs/`

## Training

HRNet-W32 + full FGE:

```bash
bash FreqPose/scripts/train_exlpose_fge_hrnet.sh
```

ResNet-50 + full FGE:

```bash
bash FreqPose/scripts/train_exlpose_fge_res50.sh
```

## Evaluation

GT-box ExLPose split evaluation:

```bash
bash FreqPose/scripts/eval_exlpose_fge_splits.sh checkpoints/hrnet_fge_best.pth hrnet
```

GT-box ExLPose + OCN evaluation:

```bash
bash FreqPose/scripts/eval_exlpose_fge_all.sh checkpoints/hrnet_fge_best.pth hrnet
```

Detector-box evaluation:

```bash
bash FreqPose/scripts/eval_exlpose_fge_all_detector.sh checkpoints/hrnet_fge_best.pth hrnet
```

## Detector Pipeline

Prepare and train the YOLO detector:

```bash
bash FreqPose/scripts/train_yolo_exlpose_det.sh
```

Generate detector-box JSON files:

```bash
python FreqPose/scripts/gen_yolo_bbox_json.py
```

## TensorBoard

Launch TensorBoard for training logs:

```bash
bash FreqPose/scripts/tb_fge.sh
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
- [ExLPose](https://cg.postech.ac.kr/research/ExLPose/) for the dataset and the benchmark used by this project.
