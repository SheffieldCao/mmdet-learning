_base_ = [ './mask_rcnn_gpwin_sfp_cs.py']

# model settings
model = dict(
    backbone=dict(
        type='GPWin',
        window_size=16,
        depths=(3, 3, 3, 3),
        num_heads=(12, 12, 12, 12),
        drop_rate=0.3,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        gp_norm_cfg=dict(type='LN'),
    ),
)

## fintune model series URL
# load_from='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa
# load_from='~/mmdet/outputs/pretrained_models/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa

## data
# batch size
data = dict(
    samples_per_gpu=2,
)

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
cudnn_benchmark = False

# resume
# resume_from = '~/mmdet/outputs/mask_rcnn_x50_32x4d_dw_gn_cs_2x2_cs_1024_from_scratch/epoch_9.pth'