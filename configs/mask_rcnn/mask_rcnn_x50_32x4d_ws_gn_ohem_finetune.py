_base_ = [ './mask_rcnn_r50_fpn_cs_8x2.py']

# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=None
    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='OHEMSampler',
                )
            ),
        )
)

## fintune model series URL: <https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws>_.
# load_from='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa
load_from='~/mmdet/outputs/mask_rcnn_x50_32x4d_ws_gn_cs_8x2_cs_1024/epoch_29.pth' #noqa

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    gamma=0.2,
    step=[15, 26])
runner = dict(type='EpochBasedRunner', max_epochs=30)
cudnn_benchmark = False

# resume
# resume_from = '~/mmdet/outputs/mask_rcnn_x50_32x4d_ws_gn_ohem_finetune_cs_4x2_1024_20e/epoch_24.pth'