_base_ = [ 
    '../_base_/models/depth_r50_fpn.py',
    '../_base_/default_runtime.py'
]

# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://jhu/resnext50_32x4d_gn_ws')
    ),
    neck=dict(
        type='FPN',
        out_channels=256,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        num_outs=4),
    depth_head=dict(
        type='SimpleDepthHead',
        in_channels=[256, 256, 256],
        num_outs=5,
        loss_depth=dict(
            type='SILogLoss', variance_focus=0.85, multi_scale_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            scale_factor=10, loss_weight=1.0),
        ffn_act_cfg=dict(type='GELU'),
        act_cfg=dict(type='GELU'),
    )
)

load_from: None

# dataset settings
img_h, img_w = 512, 1024
dataset_type = 'CityscapesDVISDataset'
data_root = 'cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.5),
    dict(
        type='Resize', img_scale=(img_w, img_h), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_depth']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations_dvis_mmdet_cvt/dvis_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bitDVIS/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations_dvis_mmdet_cvt/dvis_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bitDVIS/val/',
        pipeline=test_pipeline),
)
evaluation = dict(metric=['depth'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
runner = dict(type='EpochBasedRunner', max_epochs=55)
cudnn_benchmark = False