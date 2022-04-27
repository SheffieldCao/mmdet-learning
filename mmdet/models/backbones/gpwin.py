import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.utils.weight_init import trunc_normal_, constant_init
from mmcv.runner import BaseModule
from mmcv.utils import to_2tuple
from mmcv.cnn import build_activation_layer

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.transformer import nlc_to_nchw, nchw_to_nlc, PatchEmbed
from .swin import SwinBlock
from .resnet import BasicBlock, Bottleneck


class BasicBlockv2(BasicBlock):
    """Reimplements of BasicBlock (supported ` GELU ` etc.) in ResNet.
    """
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlockv2, self).__init__(inplanes, planes, stride, dilation, downsample, style, 
                                           with_cp, conv_cfg, norm_cfg, dcn, plugins, init_cfg)
        self.act_cfg = act_cfg
        if self.act_cfg is not None:
            if getattr(self.act_cfg,'type', None) == 'GELU':
                assert 'in_place' not in self.act_cfg, " `GELU ` got an unexpected argument ` in_place `"
            self.relu = build_activation_layer(self.act_cfg)
        
        assert getattr(self.norm_cfg, 'type', None) != 'LN', " ` norm_cfg ` not supports ` LN `"

        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        logger.warn(f'The last layer of {self.__class__.__name__} is initialized by zeros.')
        super(BasicBlock, self).init_weights()

        constant_init(self.conv2, val=0)

class Bottleneckv2(Bottleneck):
    """Reimplements of Bottleneck (supported ` GELU ` etc.) in ResNet.
    """
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 dcn=None,
                 plugins=None,
                 gp_stem_zero=False,
                 init_cfg=None):
        super(Bottleneckv2, self).__init__(inplanes, planes, stride, dilation, downsample, style, with_cp,
                                            conv_cfg, norm_cfg, dcn, plugins, init_cfg)
        self.act_cfg = act_cfg
        self.zero_gp_stem = gp_stem_zero
        if self.act_cfg is not None:
            if getattr(self.act_cfg,'type', None) == 'GELU':
                assert 'in_place' not in self.act_cfg, " `GELU ` got an unexpected argument ` in_place `"
            self.relu = build_activation_layer(self.act_cfg)

        assert getattr(self.norm_cfg, 'type', None) != 'LN', " ` norm_cfg ` not supports ` LN `"

        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        logger.warn(f'The last layer of {self.__class__.__name__} is initialized by zeros.')
        super(Bottleneckv2, self).init_weights()
        
        if self.zero_gp_stem:
            constant_init(self.conv3, val=.0)

class GPWinBlockSequence(BaseModule):
    """Implements one stage in GPWin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        gp_conv_cfg (dict, optional): The config dict of activation function.
            Default: None.
        gp_norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='BN'),
        gp_act_cfg (dict, optional): The config dict of activation.
            Default: dict(type='GELU'),
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        with_shift_block (bool, optional): Use shift block or not.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 gp_conv_cfg=None,
                 gp_norm_cfg=dict(type='BN'),
                 gp_act_cfg=dict(type='GELU'),
                 gp_stem_zero=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 with_shift_block=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        shift = False
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0 and with_shift_block:
                shift = True
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=shift,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)
        
        assert embed_dims % 4 == 0, "` embed_dims ` must be a multiple of 4"

        self.global_propagation = Bottleneckv2(
            embed_dims, embed_dims//4, with_cp=with_cp, conv_cfg=gp_conv_cfg, norm_cfg=gp_norm_cfg, 
            act_cfg=gp_act_cfg, gp_stem_zero=gp_stem_zero
            )


    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        
        x = self.global_propagation(nlc_to_nchw(x, hw_shape))
        
        assert (x.size()[2], x.size()[3]) == hw_shape, "ResBlock output shape is not equal to input shape"
        x = nchw_to_nlc(x)

        return x, hw_shape


@BACKBONES.register_module()
class GPWin(BaseModule):
    """Implements of GPWin Transformer.

    This is an implementation of paper `Exploring Plain Vision Transformer 
    Backbones for Object Detection <https://arxiv.org/abs/2203.16527>`_.

    Args:
        pretrain_img_size (int | tuple, optional): The input image size. Default: 224.
        in_channels (int, optional): The input channels. Default: 3.
        embed_dims (int, optional): The feature dimension. Default: 768.
        patch_size (int, optional): The patch size. Default: 16.
        window_size (int, optional): The local window scale. Default: 16.
        mlp_ratios (int, optional): The ratio of hidden layers in MLP. Default: 4.
        depth (list[int], optional): The number of layers in each Transformer stage. Default: (3, 3, 3, 3).
        num_heads (list[int], optional): Parallel attention heads. Default: (12, 12, 12, 12).
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool, optional): Whether to use patch normalization.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        use_abs_pos_embed (bool, optional): Whether to use absolute position embedding. Default: False.
        gp_conv_cfg (dict, optional): The config dict of activation function.
            Default: None.
        gp_norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='BN'),
        gp_act_cfg (dict, optional): The config dict of activation.
            Default: dict(type='GELU'),
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        with_shift_block (bool, optional): Whether to use shift block. Default: False.
        pretrained (str, optional): (Deprecated) Pre-trained model directory. 
            Default: None.
        convert_weights (bool, optional): Whether to convert the weights to swin format. 
            Default: False.
        frozen_stages (int, optional): Num of Stages to be frozen (stop grad and set eval mode). 
            Default: -1, frozen non-stage.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 patch_size=16,
                 window_size=16,
                 mlp_ratio=4,
                 depths=(3, 3, 3, 3),
                 num_heads=(12, 12, 12, 12),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 gp_conv_cfg=dict(type='ConvWS'),
                 gp_norm_cfg=dict(type='BN'),
                 gp_act_cfg=dict(type='GELU'),
                 gp_stem_zero=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 with_shift_block=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(GPWin, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.use_abs_pos_embed = use_abs_pos_embed

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = nn.ModuleList()
        for i in range(num_layers):
            stage = GPWinBlockSequence(
                embed_dims=embed_dims,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * embed_dims,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                gp_conv_cfg=gp_conv_cfg,
                gp_norm_cfg=gp_norm_cfg,
                gp_act_cfg=gp_act_cfg,
                gp_stem_zero=gp_stem_zero,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                with_shift_block=with_shift_block,
                init_cfg=None)
            self.stages.append(stage)

        self.num_features = embed_dims
        layer = build_norm_layer(norm_cfg, self.num_features)[1]
        self.add_module(f'final_norm', layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GPWin, self).train(mode)
        # without using _freeze_stages, the last layer norm was shut down in this function!
        # self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        
        # only remain the last norm for output layer
        norm_layer = getattr(self, f'final_norm')
        norm_layer.eval()
        for param in norm_layer.parameters():
            param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        # TODO: now ONLY support training from scratch
        logger.warn(f'No pre-trained weights for '
                    f'{self.__class__.__name__}, '
                    f'training start from scratch')
        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1.0)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for stage in self.stages:
            x, hw_shape = stage(x, hw_shape)


        norm_layer = getattr(self, f'final_norm')
        out = norm_layer(x)
        out = out.view(-1, *hw_shape, 
                    self.num_features).permute(0, 3, 1, 2).contiguous()

        return (out,)