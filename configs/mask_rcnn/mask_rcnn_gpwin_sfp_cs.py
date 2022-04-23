_base_ = [
    '../_base_/models/mask_rcnn_gpwin12_sfp.py',
    '../_base_/default_runtime.py'
]

# model setting
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        type='GPWin',
        window_size=16,
        depths=(3, 3, 3, 3),
        num_heads=(12, 12, 12, 12),
        drop_rate=0.3,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        gp_conv_cfg=conv_cfg,
        gp_norm_cfg=norm_cfg,
    ),
    neck=dict(
        type='SFP',
        out_channels=256,
        num_outs=4,
        strides=[2,1,1/2,1/4],
        conv_cfg=conv_cfg,
        norm_cfg=dict(type='LN'),
        ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            num_classes=8,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg),
        mask_head=dict(
            num_classes=8,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg))
    )
find_unused_parameters=True
load_from=None

# dataset settings
img_h, img_w = 512, 1024
# img_h, img_w = 1024, 2048
dataset_type = 'CityscapesDataset'
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
        contrast_range=(0.1, 2.0),
        saturation_range=(0.1, 2.0),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
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
            'annotations_mmdet_cvt/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations_mmdet_cvt/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations_mmdet_cvt/instancesonly_filtered_gtFine_test.json',
        img_prefix=data_root + 'leftImg8bit/test/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.05)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52],
    gamma=0.3,
    )
runner = dict(type='EpochBasedRunner', max_epochs=55)
cudnn_benchmark = True