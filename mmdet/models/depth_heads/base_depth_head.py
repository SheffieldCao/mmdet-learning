from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import FFN, nlc_to_nchw, nchw_to_nlc
from mmcv.cnn.utils.weight_init import constant_init, kaiming_init
from mmcv.runner import BaseModule, force_fp32
from ..builder import HEADS, build_loss

class CSABlock(BaseModule):
    '''Cross Scale Attention Block'''
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 init_cfg=None):
        super(CSABlock, self).__init__(init_cfg)
        # mlp

class MSRBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 init_cfg=None):
        super(MSRBlock, self).__init__(init_cfg)
        # mlp
        # self.ffn = FFN(
        #     embed_dims=embed_dims,
        #     feedforward_channels=feedforward_channels,
        #     num_fcs=2,
        #     ffn_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     act_cfg=act_cfg,
        #     add_identity=True,
        #     init_cfg=None)


@HEADS.register_module()
class SimpleDepthHead(BaseModule):
       
    def __init__(self,
                 in_channels,
                 num_ins,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', scale_factor=2),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SimpleDepthHead, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.upsample_cfg = upsample_cfg.copy()

        assert self.num_ins == self.num_outs - 2

        self.depth_head_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()

        for i in range(self.num_outs):
            if i == self.num_outs-1:
                f_conv = None
            else:
                f_conv = FFN(self.in_channels, 4*self.in_channels, act_cfg=act_cfg, init_cfg=init_cfg)
            d_conv = ConvModule(
                self.in_channels,
                1,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.depth_head_convs.append(d_conv)
            self.fuse_convs.append(f_conv)

    def init_weights(self):
        super(SimpleDepthHead, self).init_weights()
        for m in [self.depth_head_convs, self.fuse_convs]:
            if m is None:
                continue
            elif hasattr(m, 'init_weights'):
                m.init_weights()
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    # async def async_simple_test(self,
    #                             x,
    #                             proposal_list,
    #                             img_metas,
    #                             proposals=None,
    #                             rescale=False):
    #     """Async test without augmentation."""
    #     assert self.with_bbox, 'Bbox head must be implemented.'

    #     det_bboxes, det_labels = await self.async_test_bboxes(
    #         x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
    #     bbox_results = bbox2result(det_bboxes, det_labels,
    #                                self.bbox_head.num_classes)
    #     if not self.with_mask:
    #         return bbox_results
    #     else:
    #         segm_results = await self.async_test_mask(
    #             x,
    #             img_metas,
    #             det_bboxes,
    #             det_labels,
    #             rescale=rescale,
    #             mask_test_cfg=self.test_cfg.get('mask'))
    #         return bbox_results, segm_results

    def aug_test(self, x, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_ins
        inputs = [None, None] + inputs

        outs = []
        # build heads
        up_flow = input[-1]
        for i in range(self.num_outs):
            # Fuse
            if i == 0:
                pass
            else:
                assert "scale_factor" in self.upsample_cfg
                up_flow = F.interpolate(up_flow, **self.upsample_cfg)
                if inputs[4-i] is not None:
                    up_flow = self.fuse_convs[4-i](inputs[4-i]+up_flow)
                else:
                    up_flow = self.fuse_convs[4-i](up_flow)
            
            # Head
            outs.append(self.depth_head_convs[4-i](up_flow))

        return tuple(outs)

@HEADS.register_module()
class BaseDepthHead(BaseModule, metaclass=ABCMeta):
    """Base class for depth heads."""

    def __init__(self, init_cfg):
        super(BaseDepthHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def get_results(self, **kwargs):
        """Get precessed :obj:`DepthData` of multiple images."""
        pass

    def forward_train(self,
                      x,
                      gts,
                      img_metas,
                      **kwargs):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            gts (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        outs = self(x)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'
        loss = self.loss(
            *outs,
            gt_labels=gts,
            img_metas=img_metas,
            **kwargs)
        return loss

    def simple_test(self,
                    feats,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`DepthData`]: Relative depth map \
                results of each image after the post process. \
        """
        outs = self(feats)

        mask_inputs = outs + (img_metas, )
        results_list = self.get_results(
            *mask_inputs,
            rescale=rescale,
            **kwargs)
        return results_list

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')