from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import constant_init, kaiming_init
from mmcv.runner import BaseModule, force_fp32
from ..builder import HEADS, build_loss
from ..utils import interpolate_as, nlc_to_nchw, nchw_to_nlc


@HEADS.register_module()
class SimpleDepthHead(BaseModule):
       
    def __init__(self,
                 in_channels,
                 num_ins,
                 num_outs,
                 loss_depth=dict(
                     type='SILogLoss',
                     multi_scale=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 ffn_act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', scale_factor=2),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SimpleDepthHead, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg.copy()
        self.loss_depth = build_loss(loss_depth)

        assert self.num_ins == self.num_outs - 2

        self.depth_head_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()

        for i in range(self.num_outs):
            if i == self.num_outs-1:
                f_conv = None
            else:
                f_conv = FFN(self.in_channels[0], 4*self.in_channels[0], act_cfg=act_cfg, init_cfg=init_cfg)
            d_conv = nn.Sequential(
                ConvModule(self.in_channels[0],
                                1,
                                3,
                                padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                inplace=False),
                nn.Sigmoid()
            )

            self.depth_head_convs.append(d_conv)
            self.fuse_convs.append(f_conv)

    # def init_weights(self):
    #     super(SimpleDepthHead, self).init_weights()
    #     for m in [self.depth_head_convs, self.fuse_convs, self.depth_header]:
    #         if m is None:
    #             continue
    #         elif hasattr(m, 'init_weights'):
    #             m.init_weights()
    #         elif hasattr(m, 'weight') and hasattr(m, 'bias'):
    #             nn.init.kaiming_normal_(
    #                 m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.constant_(m.bias, 0)
    #         else:
    #             raise NotImplementedError("Don't know how to init {}".format(m))

    # @force_fp32(apply_to=('depth_preds', ))
    def loss(self, depth_preds, gt_depth):
        """Get the multi scale loss of depth head.

        Args:
            depth_preds (list, [Tensor,]): The input logits with the shape (N, C, H, W).
            gt_depth: The ground truth of depth estimation with the shape (N, H, W).

        Returns:
            dict: the loss of depth head.
        """
        multi_scale_loss_weights = self.loss_depth.multi_scale_weight
        assert isinstance(multi_scale_loss_weights, list) and isinstance(depth_preds, list)
        assert len(multi_scale_loss_weights) == len(depth_preds)

        loss = 0
        for i,scale_loss_weight in enumerate(multi_scale_loss_weights):
            depth_pred = depth_preds[i]
            if depth_pred.shape[-2:] != gt_depth.shape[-2:]:
                depth_pred = interpolate_as(depth_pred, gt_depth)
            depth_pred = depth_pred.permute((0, 2, 3, 1))
            assert depth_pred.size()[-1] == 1, "depth_pred channel != 1"

            # cal depth loss
            loss += scale_loss_weight*self.loss_depth(
                                        depth_pred[..., 0],  # [N, H, W, C] => [N, H, W]
                                        gt_depth)
        return dict(loss_depth=loss)

    def simple_test(self, x, img_metas, rescale=False):
        output = self.forward(x)
        depth_preds = output['depth_preds']
        if isinstance(depth_preds, list):
            depth_preds = depth_preds[-1]
        depth_preds = F.interpolate(
            depth_preds,
            size=img_metas[0]['pad_shape'][:2],
            mode='bilinear',
            align_corners=False)

        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            depth_preds = depth_preds[:, :, :h, :w]

            h, w, _ = img_metas[0]['ori_shape']
            depth_preds = F.interpolate(
                depth_preds, size=(h, w), mode='bilinear', align_corners=False)
        assert depth_preds.size()[1] == 1, "depth_pred channel != 1"
        return [depth_preds[i,0,...] for i in range(depth_preds.size()[0])]

    def aug_test(self, x, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        raise NotImplementedError("`aug_test` not implemented")

    def forward_dummy(self, x):
        """Dummy forward function.
        Used for computing network flops.
        """
        # depth head
        outs = ()
        depth_preds = self.forward(x)
        outs += (depth_preds["depth_preds"],)
        return outs

    def forward_train(self, x, gt_depth):
        output = self.forward(x)
        depth_preds = output['depth_preds']
        return self.loss(depth_preds, gt_depth)

    def forward(self, inputs):
        """Forward function."""
        outs = []
        # build heads
        up_flow = inputs[-1]
        for i in range(self.num_outs):
            # Fuse
            if i == 0:
                pass
            else:
                assert "scale_factor" in self.upsample_cfg
                up_flow = F.interpolate(up_flow, **self.upsample_cfg)
                hw_shape = up_flow.size()[-2:]
                if i < 3:
                    up_flow = nlc_to_nchw(self.fuse_convs[4-i](nchw_to_nlc(inputs[3-i]+up_flow)), hw_shape)
                else:
                    up_flow = nlc_to_nchw(self.fuse_convs[4-i](nchw_to_nlc(up_flow)), hw_shape)
                assert up_flow.size()[-2:] == hw_shape, "Fuse Mlp output shape {0} != input shape {1}".format(up_flow.size(),hw_shape)
            
            # Head
            outs.append(self.depth_head_convs[4-i](up_flow))

        return dict(depth_preds=outs)

# class CSABlock(BaseModule):
#     '''Cross Scale Attention Block'''
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  window_size,
#                  init_cfg=None):
#         super(CSABlock, self).__init__(init_cfg)
#         # mlp

# class MSRBlock(BaseModule):
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  window_size,
#                  init_cfg=None):
#         super(MSRBlock, self).__init__(init_cfg)
#         # mlp
#         # self.ffn = FFN(
#         #     embed_dims=embed_dims,
#         #     feedforward_channels=feedforward_channels,
#         #     num_fcs=2,
#         #     ffn_drop=drop_rate,
#         #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
#         #     act_cfg=act_cfg,
#         #     add_identity=True,
#         #     init_cfg=None)


# @HEADS.register_module()
# class BaseDepthHead(BaseModule, metaclass=ABCMeta):
#     """Base class for depth heads."""

#     def __init__(self, init_cfg):
#         super(BaseDepthHead, self).__init__(init_cfg)

#     @abstractmethod
#     def loss(self, **kwargs):
#         pass

#     @abstractmethod
#     def get_results(self, **kwargs):
#         """Get precessed :obj:`DepthData` of multiple images."""
#         pass

#     def forward_train(self,
#                       x,
#                       gts,
#                       img_metas,
#                       **kwargs):
#         """
#         Args:
#             x (list[Tensor] | tuple[Tensor]): Features from FPN.
#                 Each has a shape (B, C, H, W).
#             gts (list[Tensor]): Ground truth labels of all images.
#                 each has a shape (num_gts,).
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         outs = self(x)

#         assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
#                                         'even if only one item is returned'
#         loss = self.loss(
#             *outs,
#             gt_labels=gts,
#             img_metas=img_metas,
#             **kwargs)
#         return loss

#     def simple_test(self,
#                     feats,
#                     img_metas,
#                     rescale=False,
#                     **kwargs):
#         """Test function without test-time augmentation.

#         Args:
#             feats (tuple[torch.Tensor]): Multi-level features from the
#                 upstream network, each is a 4D-tensor.
#             img_metas (list[dict]): List of image information.
#             rescale (bool, optional): Whether to rescale the results.
#                 Defaults to False.

#         Returns:
#             list[obj:`DepthData`]: Relative depth map \
#                 results of each image after the post process. \
#         """
#         outs = self(feats)

#         mask_inputs = outs + (img_metas, )
#         results_list = self.get_results(
#             *mask_inputs,
#             rescale=rescale,
#             **kwargs)
#         return results_list

#     def onnx_export(self, img, img_metas):
#         raise NotImplementedError(f'{self.__class__.__name__} does '
#                                   f'not support ONNX EXPORT')