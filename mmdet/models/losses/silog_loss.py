import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class SILogLoss(nn.Module):
    '''SILog Loss for depth estimation.

    Args:
        variance_focus (float, optional) [0, 1]: Depth error variance term weight.
            Default: 0.85.
        scale_factor (float, optional): Scale factor of sqrt(D(g)). 
            Default: 10.
        loss_weight (float, optional): If True, use linear scale of loss instead of
            log scale. Default: 1.0.

    '''
    def __init__(self, 
                 variance_focus,
                 multi_scale_weight,
                 scale_factor=10,
                 loss_weight=1.0):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.multi_scale_weight = multi_scale_weight
        self.loss_weight = loss_weight
        self.scale_factor = scale_factor

    def forward(self, depth_est, depth_gt, multi_scale_loss_weight):
        assert isinstance(depth_est, torch.Tensor), "Depth forward outputs item type must be `torch.Tensor` ."
        mask = depth_gt > 1.0
        mask = mask.to(torch.bool)

        log_dist = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        D = (log_dist ** 2).mean() - self.variance_focus * (log_dist.mean() ** 2)

        return self.loss_weight * multi_scale_loss_weight * self.scale_factor * torch.sqrt(D)
