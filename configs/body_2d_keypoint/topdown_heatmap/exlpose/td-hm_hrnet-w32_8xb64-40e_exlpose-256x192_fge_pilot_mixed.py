_base_ = ['../../../_base_/default_runtime.py']
import os

metafile = 'configs/_base_/datasets/exlpose.py'
data_root = os.getenv('EXLPOSE_DATA_ROOT', 'data/ExLPose/')
dataset_type = 'ExlposeDataset'
data_mode = 'topdown'
work_dir = 'work_dirs/tdhm_hrnet_w32_exlpose_fge_pilot40_mixed'

train_cfg = dict(max_epochs=40, val_interval=2)

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    lowlight_module=dict(
        type='LowLightFGE',
        levels=4,
        local_strength=0.35,
        assume_input_normed=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        input_range=255.0),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=14,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

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

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))
param_scheduler = [
    dict(type='LinearLR', begin=0, end=80, start_factor=0.01, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        milestones=[28, 36],
        gamma=0.1,
        by_epoch=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=100))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file=metafile),
        ann_file='Annotations/ExLPose_train_LL_WL_merged.json',
        data_prefix=dict(img='ExLPose/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file=metafile),
        ann_file='Annotations/ExLPose_test_LL-A.json',
        data_prefix=dict(img='ExLPose/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'Annotations/ExLPose_test_LL-A.json'))
test_evaluator = val_evaluator
