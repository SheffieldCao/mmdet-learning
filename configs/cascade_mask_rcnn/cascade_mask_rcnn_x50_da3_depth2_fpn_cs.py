_base_ = [ './cascade_mask_rcnn_r50_fpn_cs.py']

# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNeXt_',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://jhu/resnext50_32x4d_gn_ws')
    ),
    neck=dict(
        type='PADepthAwareFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        depth_aware_levels=[2]
        ),
)

## fintune model series URL: <https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws>_.
# load_from='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa
# load_from='~/mmdet/outputs/pretrained_models/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa

# data
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
    )
)

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=5e-3)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[16, 34, 40, 43],
    gamma=0.2
    )
runner = dict(type='EpochBasedRunner', max_epochs=45)
cudnn_benchmark = False

# resume
# resume_from = '~/mmdet/outputs/cascade_mask_rcnn_x50_ws_gn_cs_4x3_1024_1e-4/epoch_7.pth'