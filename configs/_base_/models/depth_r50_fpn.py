# model settings
model = dict(
    type='DVISSingleDepth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    depth_head=dict(
        type='SimpleDepthHead',
        in_channels=256,
        num_ins=3,
        num_outs=5,
        loss_depth=dict(
            type='SILogLoss', variance_focus=0.85, multi_scale_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            scale_factor=10, loss_weight=1.0),
        norm_cfg=dict(type='BN', requires_grad=True),
        upsample_cfg=dict(mode='bilinear', scale_factor=2),
        init_cfg=dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    )
    # model training and testing settings
)
