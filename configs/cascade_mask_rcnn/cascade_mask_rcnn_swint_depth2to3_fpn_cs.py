_base_ = [ './cascade_mask_rcnn_r50_fpn_cs.py']

# model settings
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer_',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='DepthAwareFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        depth_aware_levels=[2,3]
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
resume_from = '~/mmdet/outputs/cmr_swint_depth2to3_fpn_6x2_1024_1e-4/epoch_22.pth'