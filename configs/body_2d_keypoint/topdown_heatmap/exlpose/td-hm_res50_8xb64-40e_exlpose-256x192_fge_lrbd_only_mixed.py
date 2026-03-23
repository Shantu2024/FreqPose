_base_ = ['../../../_base_/default_runtime.py']
import os

# 40-epoch pilot to match HRNet pilot length.
train_cfg = dict(max_epochs=40, val_interval=40)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1.6e-4))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
metafile = 'configs/_base_/datasets/exlpose.py'
data_root = os.getenv('EXLPOSE_DATA_ROOT', 'data/ExLPose/')
dataset_type = 'ExlposeDataset'
data_mode = 'topdown'

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    lowlight_module=dict(
        type='LowLightFGE',
        levels=4,
        local_strength=0.35,
        use_glic=False,
        use_lrbd=True,
        use_dcc=False,
        hf_strength=0.35,
        assume_input_normed=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        input_range=255.0),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=14,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=300,
        start_factor=0.001,
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        milestones=[28, 36],
        gamma=0.1,
        by_epoch=True)
]

train_dataloader = dict(
    batch_size=112,
    num_workers=12,
    persistent_workers=True,
    pin_memory=False,
    prefetch_factor=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file=metafile),
        ann_file='Annotations/ExLPose_train_LL_WL_merged.json',
        data_prefix=dict(img='ExLPose/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=96,
    num_workers=6,
    persistent_workers=True,
    pin_memory=False,
    prefetch_factor=2,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file=metafile),
        ann_file='Annotations/ExLPose_test_LL-A.json',
        data_prefix=dict(img='ExLPose/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'Annotations/ExLPose_test_LL-A.json'),
    score_mode='bbox')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = 'work_dirs/tdhm_res50_exlpose_fge_lrbd_only_mixed'
