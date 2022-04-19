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
    )
)

## fintune model series URL: <https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws>_.
# load_from='https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa
# load_from='~/mmdet/outputs/pretrained_models/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth' #noqa

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=5e-4)
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

# resume
# resume_from = '~/mmdet/outputs/mask_rcnn_x50_32x4d_dw_gn_cs_2x2_cs_1024_from_scratch/epoch_9.pth'