# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .dvis_two_stage import DVISTwoStageDetector


@DETECTORS.register_module()
class DVISMaskRCNN(DVISTwoStageDetector):
    """Implementation of DVIS Mask R-CNN """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 depth_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(DVISMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            depth_head=depth_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
