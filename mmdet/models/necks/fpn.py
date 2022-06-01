# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import build_activation_layer, build_norm_layer, build_conv_layer

from ..builder import NECKS
from ..utils.transformer import nlc_to_nchw, nchw_to_nlc
from ..depth_heads import ResnetEncoder, DepthDecoder

@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class CSA(BaseModule):
    r"""CSA module implementation.
    
    Args:
        deep_channel (int): Number of input channels of low level.
        shallow_channel (int): Number of input channels of high level.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 deep_channel,
                 shallow_channel,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(CSA, self).__init__(init_cfg)
        self.deep_channel = deep_channel
        self.shallow_channel = shallow_channel
        self.out_channel = out_channel
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # linear proj
        self.linear_s = ConvModule(self.shallow_channel, self.out_channel, 1, 1, conv_cfg=conv_cfg, inplace=False)
        self.linear_d = ConvModule(self.deep_channel, self.out_channel, 1, 1, conv_cfg=conv_cfg, inplace=False)
        self.linear_d_shortcut = ConvModule(self.deep_channel, self.shallow_channel, 1, 1, conv_cfg=conv_cfg, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, shallow, deep):
        """Forward function."""
        batch_size, _, h_s, w_s = shallow.size()
        batch_size, _, h_d, w_d = deep.size()

        x_s = self.linear_s(shallow).view(batch_size, -1, h_s*w_s).permute(0, 2, 1)
        x_d = self.linear_d(deep).view(batch_size, -1, h_d*w_d)
        attention_map = self.softmax(torch.bmm(x_s, x_d))

        x_d_shortcut = self.linear_d_shortcut(deep).view(batch_size, -1, h_d*w_d)
        x_csa = torch.bmm(x_d_shortcut, attention_map.permute(0, 2, 1)).view(batch_size, -1, h_s, w_s)
        # out = self.alpha * x_csa + x_s.permute(0, 2, 1).view(batch_size, -1, h_s, w_s)
        out = x_csa + shallow
        return out

@NECKS.register_module()
class CSAFPN(BaseModule):
    r"""CSA Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CSAFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        # self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.csa_modules = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            # self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            # add csa module
            if i <= self.backbone_end_level - 2:
                csa_block = CSA(
                    in_channels[i+1], in_channels[i], out_channels//4, 
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if self.no_norm_on_lateral else None,
                    )
                self.csa_modules.append(csa_block)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = list(inputs[self.start_level:self.backbone_end_level])

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.csa_modules[i - 1](laterals[i - 1], laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class PA(BaseModule):
    r'''Single Head Self Position Attention Module.
    Refer to `DANet`. 
    '''
    def __init__(self,
                 in_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PA, self).__init__(init_cfg)
        self.in_channel = in_channel
        # self.out_channel = out_channel
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # linear proj
        self.linear_q = ConvModule(self.in_channel, self.in_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_k = ConvModule(self.in_channel, self.in_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_v = ConvModule(self.in_channel, self.in_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.short_cut = ConvModule(self.in_channel, self.in_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=conv_cfg, inplace=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Forward function."""
        batch_size, _, h_x, w_x = x.size()

        x_k = self.linear_k(x).view(batch_size, -1, h_x*w_x).permute(0, 2, 1)
        x_q = self.linear_q(x).view(batch_size, -1, h_x*w_x)
        attention_map = self.softmax(torch.bmm(x_k, x_q))

        x_v = self.linear_v(x).view(batch_size, -1, h_x*w_x)
        x_pa = torch.bmm(x_v, attention_map.permute(0, 2, 1)).view(batch_size, -1, h_x, w_x)
        out = self.alpha*x_pa + x
        return out

class AlphaHeadPA(BaseModule):
    r'''Single Head Self Position Attention Module.
    Refer to `DANet`. 
    '''
    def __init__(self,
                 in_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PA, self).__init__(init_cfg)
        self.in_channel = in_channel
        # self.out_channel = out_channel
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # linear proj
        self.linear_q = ConvModule(self.in_channel, self.in_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_k = ConvModule(self.in_channel, self.in_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_v = ConvModule(self.in_channel, self.in_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.short_cut = ConvModule(self.in_channel, self.in_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=conv_cfg, inplace=False)
        
    def forward(self, x):
        """Forward function."""
        batch_size, _, h_x, w_x = x.size()

        x_k = self.linear_k(x).view(batch_size, -1, h_x*w_x).permute(0, 2, 1)
        x_q = self.linear_q(x).view(batch_size, -1, h_x*w_x)
        attention_map = self.softmax(torch.bmm(x_k, x_q))

        x_v = self.linear_v(x).view(batch_size, -1, h_x*w_x)
        x_pa = torch.bmm(x_v, attention_map.permute(0, 2, 1)).view(batch_size, -1, h_x, w_x)
        out = self.alpha*x_pa + x
        return out

class UnpackLayerConv3d(nn.Module):
    """
    Unpacking layer with 3d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].

    Refer to: `<https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/packnet/layers01.py>_`
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 r=2, 
                 d=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type="GELU")):
        """
        Initializes a UnpackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        # self.conv = Conv2D(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
        self.conv = ConvModule(in_channels, out_channels * (r ** 2) // d, kernel_size, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the UnpackLayerConv3d layer."""
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x

@NECKS.register_module()
class DAFPN(BaseModule):
    r"""Dual Attention Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DAFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i == self.backbone_end_level - 1:
                l_conv = nn.Sequential(
                    ConvModule(in_channels[i],
                                out_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                act_cfg=act_cfg,
                                inplace=False),
                    PA(self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg))

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@NECKS.register_module()
class PAUnpackingFPN(BaseModule):
    r"""Dual Attention Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU', requires_grad=True),
                 upsample_cfg=dict(scale_factor=2),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAUnpackingFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.unpack_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i == self.backbone_end_level - 1:
                l_conv = nn.Sequential(
                    ConvModule(in_channels[i],
                                out_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                act_cfg=act_cfg,
                                inplace=False),
                    PA(self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            
            if i > 0:
                unpack_conv = UnpackLayerConv3d(out_channels, out_channels, 3, conv_cfg=conv_cfg)                
                self.unpack_convs.append(unpack_conv)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            laterals[i - 1] = laterals[i - 1] + self.unpack_convs[i-1](laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class DA(BaseModule):
    r'''Depth Aware Attention Module
    '''
    def __init__(self,
                 in_channel_d,
                 in_channel_ins,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DA, self).__init__(init_cfg)
        self.in_channel_d = in_channel_d
        self.in_channel_ins = in_channel_ins
        self.out_channel = out_channel
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # linear proj
        self.linear_d = ConvModule(self.in_channel_d, self.out_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_ins = ConvModule(self.in_channel_ins, self.out_channel//8, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.linear_v = ConvModule(self.in_channel_d, self.out_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        self.short_cut = ConvModule(self.in_channel_ins, self.out_channel, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, inplace=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, depth):
        """Forward function."""
        batch_size, _, h_x, w_x = x.size()
        _, _, h_d, w_d = depth.size()
        assert h_x == h_d and w_x == w_d, "Attention Size Do Not Match"

        x_d = self.linear_d(depth).view(batch_size, -1, h_x*w_x).permute(0, 2, 1)
        x_ins = self.linear_ins(x).view(batch_size, -1, h_x*w_x)
        attention_map = self.softmax(torch.bmm(x_d, x_ins))

        x_v = self.linear_v(depth).view(batch_size, -1, h_x*w_x)
        x_da = torch.bmm(x_v, attention_map.permute(0, 2, 1)).view(batch_size, -1, h_x, w_x)
        out = self.alpha*x_da + self.short_cut(x)
        return out

@NECKS.register_module()
class DepthAwareFPN(BaseModule):
    r"""Depth Aware Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 depth_aware_levels=[2,3],
                 depth_enc_layers=50,
                 depth_state_dict_dir='/mnt/sdf/caoxu/mmdet/models/mono_resnet50_640x192',
                 depth_enc_channels=[64,256,512,1024,2048],
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DepthAwareFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # depth model and config
        self.depth_aware_levels = depth_aware_levels
        self.depth_enc_layers = depth_enc_layers
        self.depth_state_dict_dir = depth_state_dict_dir
        self.depth_enc_channels = depth_enc_channels
        self._make_depth_model()
        self.depth_assisted_feats_channels = list(self.depth_dec.num_ch_dec[2:]) + [self.depth_enc_channels[-1]]

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i in self.depth_aware_levels:
                l_conv = DA(self.depth_assisted_feats_channels[i], in_channels[i], self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def _make_depth_model(self):
        self.depth_enc = ResnetEncoder(self.depth_enc_layers, False)
        msgs = self.depth_enc.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, "encoder.pth"), map_location='cpu'), strict=False)
        print("Loading Depth pretrained model:")
        assert len(msgs[0]) == 0, 'Missing Keys:{}'.format(msgs[0])
        print("Missing keys:{0}; unexpected keys:{1}".format(msgs[0], msgs[1]))
        self.depth_dec = DepthDecoder(self.depth_enc_channels)
        self.depth_dec.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, 'depth.pth'), map_location='cpu'))

    @auto_fp16()
    def forward(self, x):
        """Forward function."""
        # assert len(x) == 2, "Must consists of two parts `outs` and `img`"
        assert 'img' in x and 'outs' in x, "Must consists of two parts `outs` and `img`"
        inputs, img = x['outs'], x['img']
        assert len(inputs) == len(self.in_channels)
        assert len(img.size()) == 4 and img.size()[1] == 3, "Img size wrong!"

        # build depths
        depth_enc_feats = self.depth_enc(img)
        depth_dec_feats = self.depth_dec(depth_enc_feats)
        assisted_feats = []
        for stage_id in range(2, 5, 1):
            assisted_feats.append(depth_dec_feats[("dec_feat", stage_id)])
        assisted_feats.append(depth_enc_feats[-1])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level], assisted_feats[i+ self.start_level]) if i+self.start_level in self.depth_aware_levels else lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@NECKS.register_module()
class PADepthAwareFPN(BaseModule):
    r"""Self Position Attention & Depth Aware Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 depth_aware_levels=[2,3],
                 depth_enc_layers=50,
                 depth_state_dict_dir='/mnt/sdf/caoxu/mmdet/models/mono_resnet50_640x192',
                 depth_enc_channels=[64,256,512,1024,2048],
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PADepthAwareFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # depth model and config
        self.depth_aware_levels = depth_aware_levels
        self.depth_enc_layers = depth_enc_layers
        self.depth_state_dict_dir = depth_state_dict_dir
        self.depth_enc_channels = depth_enc_channels
        self._make_depth_model()
        self.depth_assisted_feats_channels = list(self.depth_dec.num_ch_dec[2:]) + [self.depth_enc_channels[-1]]

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i in self.depth_aware_levels:
                l_conv = DA(self.depth_assisted_feats_channels[i], in_channels[i], self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
            elif i == self.backbone_end_level - 1:
                l_conv = nn.Sequential(
                    ConvModule(in_channels[i],
                                out_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                act_cfg=act_cfg,
                                inplace=False),
                    PA(self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def _make_depth_model(self):
        self.depth_enc = ResnetEncoder(self.depth_enc_layers, False)
        msgs = self.depth_enc.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, "encoder.pth"), map_location='cpu'), strict=False)
        print("Loading Depth pretrained model:")
        assert len(msgs[0]) == 0, 'Missing Keys:{}'.format(msgs[0])
        print("Missing keys:{0}; unexpected keys:{1}".format(msgs[0], msgs[1]))
        self.depth_dec = DepthDecoder(self.depth_enc_channels)
        self.depth_dec.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, 'depth.pth'), map_location='cpu'))

    @auto_fp16()
    def forward(self, x):
        """Forward function."""
        # assert len(x) == 2, "Must consists of two parts `outs` and `img`"
        assert 'img' in x and 'outs' in x, "Must consists of two parts `outs` and `img`"
        inputs, img = x['outs'], x['img']
        assert len(inputs) == len(self.in_channels)
        assert len(img.size()) == 4 and img.size()[1] == 3, "Img size wrong!"

        # build depths
        depth_enc_feats = self.depth_enc(img)
        depth_dec_feats = self.depth_dec(depth_enc_feats)
        assisted_feats = []
        for stage_id in range(2, 5, 1):
            assisted_feats.append(depth_dec_feats[("dec_feat", stage_id)])
        assisted_feats.append(depth_enc_feats[-1])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level], assisted_feats[i+ self.start_level]) if i+self.start_level in self.depth_aware_levels else lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

@NECKS.register_module()
class DepthAwareCSAFPN(BaseModule):
    r"""`Depth Aware` & `Cross Scale Attention Fuse` Feature Pyramid Network.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 depth_aware_levels=[2,3],
                 depth_enc_layers=50,
                 depth_state_dict_dir='/mnt/sdf/caoxu/mmdet/models/mono_resnet50_640x192',
                 depth_enc_channels=[64,256,512,1024,2048],
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DepthAwareCSAFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # depth model and config
        self.depth_aware_levels = depth_aware_levels
        self.depth_enc_layers = depth_enc_layers
        self.depth_state_dict_dir = depth_state_dict_dir
        self.depth_enc_channels = depth_enc_channels
        self._make_depth_model()
        self.depth_assisted_feats_channels = list(self.depth_dec.num_ch_dec[2:]) + [self.depth_enc_channels[-1]]

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i in self.depth_aware_levels:
                l_conv = DA(self.depth_assisted_feats_channels[i], in_channels[i], self.out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def _make_depth_model(self):
        self.depth_enc = ResnetEncoder(self.depth_enc_layers, False)
        msgs = self.depth_enc.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, "encoder.pth"), map_location='cpu'), strict=False)
        print("Loading Depth pretrained model:")
        assert len(msgs[0]) == 0, 'Missing Keys:{}'.format(msgs[0])
        print("Missing keys:{0}; unexpected keys:{1}".format(msgs[0], msgs[1]))
        self.depth_dec = DepthDecoder(self.depth_enc_channels)
        self.depth_dec.load_state_dict(torch.load(os.path.join(self.depth_state_dict_dir, 'depth.pth'), map_location='cpu'))

    @auto_fp16()
    def forward(self, x):
        """Forward function."""
        # assert len(x) == 2, "Must consists of two parts `outs` and `img`"
        assert 'img' in x and 'outs' in x, "Must consists of two parts `outs` and `img`"
        inputs, img = x['outs'], x['img']
        assert len(inputs) == len(self.in_channels)
        assert len(img.size()) == 4 and img.size()[1] == 3, "Img size wrong!"

        # build depths
        depth_enc_feats = self.depth_enc(img)
        depth_dec_feats = self.depth_dec(depth_enc_feats)
        assisted_feats = []
        for stage_id in range(2, 5, 1):
            assisted_feats.append(depth_dec_feats[("dec_feat", stage_id)])
        assisted_feats.append(depth_enc_feats[-1])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level], assisted_feats[i+ self.start_level]) if i+self.start_level in self.depth_aware_levels else lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

# class ConvModule(nn.Module):
#     """Reimplements of a conv block that bundles conv/norm/activation layers.

#     Modified from mmcv.cnn.bricks.conv_module.ConvModule

#     Args:
#         in_channels (int): Number of channels in the input feature map.
#             Same as that in ``nn._ConvNd``.
#         out_channels (int): Number of channels produced by the convolution.
#             Same as that in ``nn._ConvNd``.
#         kernel_size (int | tuple[int]): Size of the convolving kernel.
#             Same as that in ``nn._ConvNd``.
#         stride (int | tuple[int]): Stride of the convolution.
#             Same as that in ``nn._ConvNd``.
#         padding (int | tuple[int]): Zero-padding added to both sides of
#             the input. Same as that in ``nn._ConvNd``.
#         dilation (int | tuple[int]): Spacing between kernel elements.
#             Same as that in ``nn._ConvNd``.
#         groups (int): Number of blocked connections from input channels to
#             output channels. Same as that in ``nn._ConvNd``.
#         bias (bool | str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
#             False. Default: "auto".
#         conv_cfg (dict): Config dict for convolution layer. Default: None,
#             which means using conv2d.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (dict): Config dict for activation layer.
#             Default: dict(type='ReLU').
#         inplace (bool): Whether to use inplace mode for activation.
#             Default: True.
#         padding_mode (str): If the `padding_mode` has not been supported by
#             current `Conv2d` in PyTorch, we will use our own padding layer
#             instead. Currently, we support ['zeros', 'circular'] with official
#             implementation and ['reflect'] with our own implementation.
#             Default: 'zeros'.
#         order (tuple[str]): The order of conv/norm/activation layers. It is a
#             sequence of "conv", "norm" and "act". Common examples are
#             ("conv", "norm", "act") and ("act", "conv", "norm").
#             Default: ('conv', 'norm', 'act').
#     """

#     _abbr_ = 'conv_block'

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias='auto',
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=dict(type='ReLU'),
#                  inplace=True,
#                  padding_mode='zeros',
#                  order=('conv', 'norm', 'act')):
#         super(ConvModule, self).__init__()
#         assert conv_cfg is None or isinstance(conv_cfg, dict)
#         assert norm_cfg is None or isinstance(norm_cfg, dict)
#         assert act_cfg is None or isinstance(act_cfg, dict)
#         official_padding_mode = ['zeros', 'circular']
#         assert padding_mode in official_padding_mode, "Unknown padding mode: {}".format(padding_mode)
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         self.inplace = inplace
#         self.order = order
#         assert isinstance(self.order, tuple) and len(self.order) == 3
#         assert set(order) == set(['conv', 'norm', 'act'])

#         self.with_norm = norm_cfg is not None
#         self.with_activation = act_cfg is not None
#         # if the conv layer is before a norm layer, bias is unnecessary.
#         if bias == 'auto':
#             bias = not self.with_norm
#         self.with_bias = bias

#         # build convolution layer
#         self.conv = build_conv_layer(
#             conv_cfg,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias)
#         # export the attributes of self.conv to a higher level for convenience
#         self.in_channels = self.conv.in_channels
#         self.out_channels = self.conv.out_channels
#         self.kernel_size = self.conv.kernel_size
#         self.stride = self.conv.stride
#         self.padding = padding
#         self.dilation = self.conv.dilation
#         self.transposed = self.conv.transposed
#         self.output_padding = self.conv.output_padding
#         self.groups = self.conv.groups

#         # build normalization layers
#         if self.with_norm:
#             from mmcv.utils import _BatchNorm, _InstanceNorm
#             # norm layer is after conv layer
#             if order.index('norm') > order.index('conv'):
#                 norm_channels = out_channels
#             else:
#                 norm_channels = in_channels
#             self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
#             self.add_module(self.norm_name, norm)
#             if self.with_bias:
#                 if isinstance(norm, (_BatchNorm, _InstanceNorm)):
#                     warnings.warn(
#                         'Unnecessary conv bias before batch/instance norm')
#         else:
#             self.norm_name = None

#         # build activation layer
#         if self.with_activation:
#             act_cfg_ = act_cfg.copy()
#             # nn.Tanh has no 'inplace' argument
#             if act_cfg_['type'] not in [
#                     'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
#             ]:
#                 act_cfg_.setdefault('inplace', inplace)
#             self.activate = build_activation_layer(act_cfg_)

#         # Use msra init by default
#         self.init_weights()

#     @property
#     def norm(self):
#         if self.norm_name:
#             return getattr(self, self.norm_name)
#         else:
#             return None

#     def init_weights(self):
#         # 1. It is mainly for customized conv layers with their own
#         #    initialization manners by calling their own ``init_weights()``,
#         #    and we do not want ConvModule to override the initialization.
#         # 2. For customized conv layers without their own initialization
#         #    manners (that is, they don't have their own ``init_weights()``)
#         #    and PyTorch's conv layers, they will be initialized by
#         #    this method with default ``kaiming_init``.
#         # Note: For PyTorch's conv layers, they will be overwritten by our
#         #    initialization implementation using default ``kaiming_init``.
#         from mmcv.cnn.utils.weight_init import kaiming_init, constant_init
#         if not hasattr(self.conv, 'init_weights'):
#             if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
#                 nonlinearity = 'leaky_relu'
#                 a = self.act_cfg.get('negative_slope', 0.01)
#             else:
#                 nonlinearity = 'relu'
#                 a = 0
#             kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
#         if self.with_norm:
#             constant_init(self.norm, 1, bias=0)

#     def forward(self, x):
#         for layer in self.order:
#             if layer == 'conv':
#                 x = self.conv(x)
#             elif layer == 'norm' and self.with_norm:
#                 if getattr(self.norm_cfg, 'type') == 'LN':
#                     hw_shape = x.size()[2:]
#                     x = nchw_to_nlc(x)
#                 x = self.norm(x)
#                 if getattr(self.norm_cfg, 'type') == 'LN':
#                     x = nlc_to_nchw(x, hw_shape)
#             elif layer == 'act' and self.with_activation:
#                 x = self.activate(x)
#         return x

class DeConvModule(BaseModule):
    """A deconv block bundles conv/norm/activation layers.
    
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        output_padding (int | tuple[int]): Additional size added to one 
            side of each dimension in the output shape. Default: 0
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        with_bias (bool): Default: "False".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'deconv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 with_bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU'),
                 inplace=True,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(DeConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        assert padding_mode in official_padding_mode, "only support {0}, {1} padding mode".format('zeros', 'circular')
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # if the conv layer is before a norm layer, bias is unnecessary.
        self.with_bias = with_bias
        self.with_bias = not self.with_norm

        # build convolution layer
        self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, 
                groups=groups, bias=self.with_bias, dilation=dilation, padding_mode=padding_mode)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            from mmcv.utils import _BatchNorm, _InstanceNorm
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        from mmcv.cnn.utils.weight_init import kaiming_init, constant_init
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                if getattr(self.norm_cfg, 'type') == 'LN':
                    hw_shape = x.size()[2:]
                    x = nchw_to_nlc(x)
                x = self.norm(x)
                if getattr(self.norm_cfg, 'type') == 'LN':
                    x = nlc_to_nchw(x, hw_shape)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x


@NECKS.register_module()
class SFP(BaseModule):
    r"""Simple Feature Pyramid Neck.

    This is an implementation of paper `Exploring Plain Vision Transformer 
    Backbones for Object Detection <https://arxiv.org/abs/2203.16527>`_.

    Args:
        in_channel (int): Number of input channel of the last scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        strides (tuple[int]): Conv strides.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        deconv_norm_cfg (dict): Config dict for deconvolution block normalization layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 in_channel,
                 out_channels,
                 num_outs,
                 strides=(1/4,1/2,1,2,4),
                 conv_cfg=None,
                 deconv_norm_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SFP, self).__init__(init_cfg)
        assert isinstance(in_channel, int)
        self.in_channel = in_channel
        self.out_channels = out_channels
        assert len(strides) == num_outs
        self.lateral_convs = nn.ModuleList()

        for stride in strides:
            if stride == 4:
                l_conv = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvModule(self.in_channel, self.out_channels, 1, conv_cfg=conv_cfg, act_cfg=None, inplace=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvModule(self.out_channels, self.out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, inplace=False),
                )           
            elif stride == 2:
                l_conv = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvModule(self.in_channel, self.out_channels, 1, conv_cfg=conv_cfg, act_cfg=None, inplace=False),
                    ConvModule(self.out_channels, self.out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, inplace=False),
                )
            elif stride == 1:
                l_conv = nn.Sequential(
                    ConvModule(self.in_channel, self.out_channels, 1, conv_cfg=conv_cfg, act_cfg=None, inplace=False),
                    ConvModule(self.out_channels, self.out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, inplace=False),
                )
            elif stride == 1/2:
                l_conv = nn.Sequential(
                    DeConvModule(self.in_channel, self.in_channel, kernel_size=3, stride=2, padding=1, output_padding=1, conv_cfg=conv_cfg, 
                        norm_cfg=deconv_norm_cfg, act_cfg=act_cfg, inplace=False),
                    ConvModule(self.in_channel, self.out_channels, 1, conv_cfg=conv_cfg, act_cfg=None, inplace=False),
                    ConvModule(self.out_channels, self.out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, inplace=False),
                )
            elif stride == 1/4:
                l_conv = nn.Sequential(
                    DeConvModule(self.in_channel, self.in_channel, kernel_size=3, stride=2, padding=1, output_padding=1, conv_cfg=conv_cfg, 
                        norm_cfg=deconv_norm_cfg, act_cfg=act_cfg, inplace=False),
                    ConvModule(self.in_channel, self.out_channels, 1, conv_cfg=conv_cfg, act_cfg=None, inplace=False),
                    DeConvModule(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, conv_cfg=conv_cfg, 
                        norm_cfg=deconv_norm_cfg, act_cfg=act_cfg, inplace=False),
                    ConvModule(self.out_channels, self.out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None, inplace=False),
                )         

            self.lateral_convs.append(l_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, tuple)

        # build outputs(top down results)
        outs = []
        for lateral_conv in self.lateral_convs:
            outs.append(lateral_conv(inputs[0])) 

        return tuple(outs)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, 3, 2, 1, output_padding=1, 
                    groups=1, dilation=1, padding_mode=padding_mode
                    )
        self.norm = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        return self.act(x)

class Upsample(nn.Module):
    """
    Upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        padding_mode (str): Padding mode.
    """

    def __init__(self, scale_factor, in_channels, out_channels, padding_mode):
        super(Upsample, self).__init__()

        if scale_factor == 1/2:
            self.conv_norm_act = UpBlock(in_channels, in_channels, padding_mode)
        elif scale_factor == 1/4:
            self.conv_norm_act = nn.Sequential(OrderedDict([
                ('upconv1', UpBlock(in_channels, in_channels, padding_mode)), 
                ('upconv2', UpBlock(in_channels, in_channels, padding_mode)) 
            ]))
        elif scale_factor == 1:
            self.conv_norm_act = None
        
        self.linear = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1, 1, False, padding_mode=padding_mode)
        self.final_norm = nn.LayerNorm(out_channels, elementwise_affine=True)

    def forward(self, x):
        if self.conv_norm_act:
            x = self.conv_norm_act(x)
        
        x = self.linear(x)

        x = self.final_conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.final_norm(x).permute(0, 3, 1, 2).contiguous()
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode):
        super(DownBlock, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, 1, 1, bias=True, padding_mode=padding_mode)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return self.act(x)

class Downsample(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, padding_mode):
        super(Downsample, self).__init__()

        if scale_factor == 2:
            self.conv_norm_act = DownBlock(in_channels, in_channels, padding_mode)
        elif scale_factor == 4:
            self.conv_norm_act = nn.Sequential(OrderedDict([
                ('downconv1', DownBlock(in_channels, in_channels, padding_mode)), 
                ('downconv2', DownBlock(in_channels, in_channels, padding_mode)) 
            ]))
        else:
            raise ValueError("scale_factor must be 2 or 4")
        
        self.linear = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1, 1, False, padding_mode=padding_mode)
        self.final_norm = nn.LayerNorm(out_channels, elementwise_affine=True)

    def forward(self, x):
        if self.conv_norm_act:
            x = self.conv_norm_act(x)
        
        x = self.linear(x)

        x = self.final_conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.final_norm(x).permute(0, 3, 1, 2).contiguous()
        return x

@NECKS.register_module()
class SFPdev(BaseModule):
    r"""Simple Feature Pyramid Neck.

    This is an implementation of paper `Exploring Plain Vision Transformer 
    Backbones for Object Detection <https://arxiv.org/abs/2203.16527>`_.

    Args:
        in_channel (int): Number of input channel of the last scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        strides (list[int]): Conv strides.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 in_channel,
                 out_channels,
                 num_outs=5,
                 strides=[1/4,1/2,1,2,4],
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SFPdev, self).__init__(init_cfg)
        assert isinstance(in_channel, int)
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.num_outs = num_outs
        assert len(strides) == num_outs
        lateral_convs = OrderedDict()

        for i,stride in enumerate(strides):
            if stride <= 1:
                lateral_convs[('lateral', i)] = Upsample(stride, in_channel, out_channels, padding_mode='zeros')
            else:
                lateral_convs[('lateral', i)] = Downsample(stride, in_channel, out_channels, padding_mode='zeros')     

        self.lateral_convs = nn.ModuleList(list(lateral_convs.values()))

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, tuple)

        # build outputs(top down results)
        outs = []
        for i in range(self.num_outs):
            outs.append(self.lateral_convs[i](inputs[2])) 

        return tuple(outs)