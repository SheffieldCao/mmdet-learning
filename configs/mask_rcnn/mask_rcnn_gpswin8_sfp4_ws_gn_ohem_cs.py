_base_ = [ './mask_rcnn_gpwin_sfp_cs.py']

# model settings
model = dict(
    backbone=dict(
        type='GPWin',
        window_size=8,
        depths=(2, 2, 2, 2),
        num_heads=(12, 12, 12, 12),
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        with_shift_block=True,
        gp_stem_zero=False,
    ),
    neck=dict(
        type='SFP',
        out_channels=256,
        num_outs=4,
        strides=(1/4,1/2,1,2),
        ),
    rpn_head=dict(
        num_convs=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='OHEMSampler',
                )
            ),
        )
)

## fintune model series URL
# load_from='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa
# load_from='~/mmdet/outputs/pretrained_models/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa

## data
# batch size
data = dict(
    samples_per_gpu=2,
)

# find_unused_parameters=True

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[30, 60, 70, 75],
    gamma=0.2,
    )
runner = dict(type='EpochBasedRunner', max_epochs=80)
cudnn_benchmark = False

# resume
resume_from = '~/mmdet/outputs/mask_rcnn_gpswin8_sfp4_ws_gn_ohem_cs_4x2_1024_scratch/epoch_51.pth'