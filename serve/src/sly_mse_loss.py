# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses.mse_loss import MSELoss, mse_loss
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class SlyMSELoss(MSELoss):

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        target = torch.nan_to_num(target)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss