import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock
from mmdet.models.roi_heads.mask_heads.fused_semantic_head import FusedSemanticHead
from mmdet.models.builder import HEADS
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss


@HEADS.register_module()
class SlyFusedSemanticHead(FusedSemanticHead):
    def __init__(self, conv_to_res=False, **kwargs):
        super(SlyFusedSemanticHead, self).__init__(**kwargs)
        self.conv_to_res = conv_to_res
        if self.conv_to_res:
            num_res_blocks = self.num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                self.in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks

    @auto_fp16()
    def forward(self, feats):
        lateral_res = []
        lateral_res.append(self.lateral_convs[self.fusion_level](feats[self.fusion_level]))
        fused_size = tuple(lateral_res[0].shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                lateral_res.append(self.lateral_convs[i](feat))

        x = torch.sum(torch.stack(lateral_res), dim=0)
        for i in range(self.num_convs):
            x = self.convs[i](x)

        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        return loss_semantic_seg
