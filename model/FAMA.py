from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet
import os
import matplotlib
matplotlib.use('Agg')  # 无GUI模式

from .base.swin_transformer import SwinTransformer
from model.mymodule.ASPP import ASPP
from model.mymodule.CrossAttention import CrossMultiHeadedAttention, PositionalEncoding, zeroshot_classifier
from model.mymodule.PhaseandAmplitudeAttention import PhaseAmplitudeAttention
from generate_cam_pascal import PASCAL_CLASSES
from generate_cam_coco import COCO_CLASSES
from generate_cam_fss import FSS_CLASSES



class FAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, clip_model):
        super(FAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization  特征提取器backbone模型主干不用管
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))

        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model  定义模型
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)               # 以resnet50为例保存的是[ 3, 7, 13, 16]，用来保存不同层次特征的索引。
        self.model = FAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, clip_model=clip_model)   # 将多层特征的通道数和索引值输入网络模型中去
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.bce_entropy_loss = nn.BCEWithLogitsLoss()

    def forward(self, query_img, support_img, support_mask, class_id):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)
            # support_feats_fg = torch.zeros_like(support_img)
            # support_feats_fg[:, 0, :, :] = support_img[:, 0, :, :] * support_mask
            # support_feats_fg[:, 1, :, :] = support_img[:, 1, :, :] * support_mask
            # support_feats_fg[:, 2, :, :] = support_img[:, 2, :, :] * support_mask
            # support_feats_fg = self.extract_feats(support_feats_fg)

        # logit_mask, fg_mask = self.model(query_feats, support_feats, support_feats_fg, support_mask.clone(), class_id)
        logit_mask = self.model(query_feats, support_feats, support_mask.clone(), class_id)

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features 从backbone提取输入图像的特征 """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']
        class_id=batch['class_id']
        # support_cam = batch['support_cams'].squeeze(1)

        if nshot == 1:
            logit_mask = self(query_img, support_imgs[:, 0], support_masks[:, 0], class_id)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                n_support_feats_fg = []
                for k in range(nshot):
                    support_feats = self.extract_feats(support_imgs[:, k])
                    # support_feats_fg = torch.zeros_like(support_imgs[:, k])
                    # support_feats_fg[:, 0, :, :] = support_imgs[:, k][:, 0, :, :] * support_masks[:, k]
                    # support_feats_fg[:, 1, :, :] = support_imgs[:, k][:, 1, :, :] * support_masks[:, k]
                    # support_feats_fg[:, 2, :, :] = support_imgs[:, k][:, 2, :, :] * support_masks[:, k]
                    # support_feats_fg = self.extract_feats(support_feats_fg)
                    n_support_feats.append(support_feats)
                    # n_support_feats_fg.append(support_feats_fg)
            # logit_mask, corr_fg = self.model(query_feats, n_support_feats, n_support_feats_fg, support_masks.clone(), class_id, nshot)
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(),
                                             class_id, nshot)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def compute_objective2(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, -1)
        gt_mask = gt_mask.view(bsz, -1)

        return self.bce_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class FAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids, clip_model):
        super(FAMA_model, self).__init__()

        self.stack_ids = stack_ids
        self.high_avg_pool = nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.MaskCrossAtten_blocks = nn.ModuleList()
        self.MaskCrossAtten_blocks.append(CrossMultiHeadedAttention(h=8, d_model=48*48, dropout=0.1))
        self.MaskCrossAtten_blocks.append(CrossMultiHeadedAttention(h=8, d_model=24*24, dropout=0.1))
        self.MaskCrossAtten_blocks.append(CrossMultiHeadedAttention(h=8, d_model=12*12, dropout=0.1))
        self.PhaseAmplitudeAtten_blocks = nn.ModuleList()
        self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(512))
        self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(1024))
        self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(2048))
        # self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(256))
        # self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(512))
        # self.PhaseAmplitudeAtten_blocks.append(PhaseAmplitudeAttention(1024))
        self.clip_text_features = zeroshot_classifier(PASCAL_CLASSES, ['a photo of a {}.'], clip_model)
        # self.clip_text_features = zeroshot_classifier(COCO_CLASSES, ['a photo of a {}.'], clip_model)
        # self.clip_text_features = zeroshot_classifier(FSS_CLASSES, ['a photo of a {}.'], clip_model)
        self.linear_1024_576 = nn.Linear(1024, 128)

        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))         # pe位置编码器加上位置信息


        outch1, outch2, outch3 = 16, 64, 128         # 指定输出通道的数量
        # conv blocks
        ch = 4
        self.conv1 = self.build_conv_block((stack_ids[3] - stack_ids[2])*ch, [outch1, outch2, outch3], [3, 3, 3], [1, 1,
                                                                                                              1])  # 1/32  通过 stack_ids[3] - stack_ids[2] 确定了卷积块中第一个卷积层的输入通道数，进而构建了一个包含多个卷积层、组归一化层和 ReLU 激活函数层的卷积块空间尺度维度不会变化，通道数改变
        self.conv2 = self.build_conv_block((stack_ids[2] - stack_ids[1])*ch, [outch1, outch2, outch3], [5, 3, 3],
                                           [1, 1, 1])  # 1/16
        self.conv3 = self.build_conv_block((stack_ids[1] - stack_ids[0])*ch, [outch1, outch2, outch3], [5, 5, 3],
                                           [1, 1, 1])  # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # 统一中间特征通道数
        mid_ch = outch3 // 2

        # UNet++专用过渡卷积
        self.trans_conv1 = nn.Conv2d(outch3, mid_ch, kernel_size=1)  # x1 -> x2
        self.trans_conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=1)  # mid -> x3
        self.trans_conv3 = nn.Conv2d(outch3, mid_ch, kernel_size=1)  # x1 -> x3 (跨级)
        self.trans_conv4 = nn.Conv2d(outch3, mid_ch, kernel_size=1)  # x2 -> x3
        self.conv32_16 = self.build_conv_block(outch3 + mid_ch, [outch3, mid_ch, mid_ch], [3, 3, 3], [1, 1, 1])  # 1/32 + 1/16
        self.conv16_8 = self.build_conv_block(outch3 * 2, [outch3, outch3, outch3], [3, 3, 3],
                                              [1, 1, 1])  # 1/16 + 1/8
        self.fusion_conv1 = self.build_conv_block(outch3 + mid_ch * 3, [outch3 * 2, outch3 + mid_ch, outch3], [3, 3, 3], [1, 1, 1])
        self.ASPP_meta1 = ASPP(outch3, outch3)  # 定义一个ASPP多尺度空洞解码器融合多尺度的特征
        # mixer blocks       mixer1的第一个输入通道与前面对其
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))


        reduce_dim = outch3

        self.init_merge1 = nn.Sequential(
            nn.Conv2d(outch3 * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.beta_conv1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.init_merge2 = nn.Sequential(
            nn.Conv2d(outch3 + in_channels[1]*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.beta_conv2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.init_merge3 = nn.Sequential(
            nn.Conv2d(outch3 + in_channels[0]*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.beta_conv3 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_block = self.build_conv_block(1, [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])

        self.conv_pp = nn.Sequential(
            nn.Conv2d(in_channels[0], outch3, kernel_size=1),
            nn.BatchNorm2d(outch3),
            nn.ReLU(inplace=True)
        )
        self.query_merge3 = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1] + 2, in_channels[1], kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.support_merge3 = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1], in_channels[1], kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )


    def forward(self, query_feats, support_feats, support_mask, class_id=None, nshot=1):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue
            bsz, ch, ha, wa = query_feat.size()
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            if nshot == 1:
                support_feat = support_feats[idx]
                if idx < self.stack_ids[1]:
                    support_feat = self.PhaseAmplitudeAtten_blocks[0](support_feat)
                    query_feat = self.PhaseAmplitudeAtten_blocks[0](query_feat)
                elif idx < self.stack_ids[2]:
                    support_feat = self.PhaseAmplitudeAtten_blocks[1](support_feat)
                    query_feat = self.PhaseAmplitudeAtten_blocks[1](query_feat)
                else:
                    support_feat = self.PhaseAmplitudeAtten_blocks[2](support_feat)
                    query_feat = self.PhaseAmplitudeAtten_blocks[2](query_feat)
                mask1 = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True)
                support_feat_fg = support_feat * mask1
                Ass1 = self.qinhe(support_feat, support_feat)
                Ass2 = self.qinhe(support_feat_fg, support_feat_fg)
                Aqq = self.qinhe(query_feat, query_feat)
                Asq1 = self.qinhe(support_feat, query_feat)
                Asq2 = self.qinhe(support_feat_fg, query_feat)
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                # support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                if idx < self.stack_ids[1]:
                    query_feat = self.PhaseAmplitudeAtten_blocks[0](query_feat)
                elif idx < self.stack_ids[2]:
                    query_feat = self.PhaseAmplitudeAtten_blocks[1](query_feat)
                else:
                    query_feat = self.PhaseAmplitudeAtten_blocks[2](query_feat)

                for n in range(nshot):
                    if idx < self.stack_ids[1]:
                        support_feats[n][idx] = self.PhaseAmplitudeAtten_blocks[0](support_feats[n][idx])
                    elif idx < self.stack_ids[2]:
                        support_feats[n][idx] = self.PhaseAmplitudeAtten_blocks[1](support_feats[n][idx])
                    else:
                        support_feats[n][idx] = self.PhaseAmplitudeAtten_blocks[2](support_feats[n][idx])

                support_feats_fg = self.mask_feature2(support_feats, support_mask)
                Ass1 = [self.qinhe(support_feats[n][idx], support_feats[n][idx]) for n in range(nshot)]
                Ass2 = [self.qinhe(support_feats_fg[n][idx], support_feats_fg[n][idx]) for n in range(nshot)]
                Asq1 = [self.qinhe(support_feats[n][idx], query_feat) for n in range(nshot)]
                Aqq = self.qinhe(query_feat, query_feat)
                Asq2 = [self.qinhe(support_feats_fg[n][idx], query_feat) for n in range(nshot)]

                hw = ha * wa
                bsz = query_feat.size(0)  # 假设query_feat的batch size是bsz
                # 使用torch.stack和view组合操作
                stack_view = lambda x: torch.stack(x).permute(1, 0, 2, 3).contiguous().view(bsz, hw * nshot, hw)
                Ass1, Ass2, Asq1, Asq2 = map(stack_view, [Ass1, Ass2, Asq1, Asq2])
                # Ass1, Asq1, = map(stack_view, [Ass1, Asq1])
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                mask = mask.view(bsz, -1)
            if idx < self.stack_ids[1]:
                coarse_mask1 = self.MaskCrossAtten_blocks[0](Asq1, Ass1, Aqq, mask, nshot=1, temperature=0.1)
                coarse_mask2 = self.MaskCrossAtten_blocks[0](Asq2, Ass2, Aqq, mask, nshot=1, temperature=0.1)
            elif idx < self.stack_ids[2]:
                coarse_mask1 = self.MaskCrossAtten_blocks[1](Asq1, Ass1, Aqq, mask, nshot=1, temperature=0.1)
                coarse_mask2 = self.MaskCrossAtten_blocks[1](Asq2, Ass2, Aqq, mask, nshot=1, temperature=0.1)
            else:
                coarse_mask1 = self.MaskCrossAtten_blocks[2](Asq1, Ass1, Aqq, mask, nshot=1, temperature=0.1)
                coarse_mask2 = self.MaskCrossAtten_blocks[2](Asq2, Ass2, Aqq, mask, nshot=1, temperature=0.1)
            coarse_mask = torch.cat((coarse_mask1, coarse_mask2), dim=-1)
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, -1, ha, wa))

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3] - 1 - self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2] - self.stack_ids[0]:self.stack_ids[3] - self.stack_ids[0]]).transpose(0,1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2] - 1 - self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1] - self.stack_ids[0]:self.stack_ids[2] - self.stack_ids[0]]).transpose(0,1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1] - 1 - self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1] - self.stack_ids[0]]).transpose(0,1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        x1 = coarse_masks1
        x2 = coarse_masks2
        x3 = coarse_masks3

        # 2. 构建连接路径（添加尺寸对齐检查）
        # 2.1 第一级上采样 (1/32 -> 1/16)
        x1_up = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x1_up = self.trans_conv1(x1_up)

        # 2.2 中间级融合（修正加法操作前的维度）
        # mid_feat = self.mid_fusion(x2)  # [N,mid_ch,H/16,W/16]
        mid_feat = torch.cat([x2, x1_up], dim=1)
        mid_feat = self.conv32_16(mid_feat)  # 再次融合

        # 2.3 第二级上采样 (1/16 -> 1/8)
        mid_up = F.interpolate(mid_feat, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        mid_up = self.trans_conv2(mid_up)  # [N,mid_ch,H/8,W/8]

        # 2.4 跨级连接 (1/32 -> 1/8)
        x1_skip = F.interpolate(x1, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        x1_skip = self.trans_conv3(x1_skip)  # [N,mid_ch,H/8,W/8]

        # --- 直接连接 (x2: 1/16 -> x3: 1/8) ---
        x2_skip = F.interpolate(x2, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        x2_skip = self.trans_conv4(x2_skip)  # [N, mid_ch, H/8, W/8]

        # 3. 最终融合（修正通道拼接）
        mix = torch.cat([x3, mid_up, x1_skip, x2_skip], dim=1)  # [N, outch3+mid_ch*2, H/8, W/8]
        mix = F.layer_norm(mix, mix.shape[-3:])
        mix = self.fusion_conv1(mix)

        with torch.no_grad():
            # Embedding clip text features使用clip获取类别特征提高模型的泛化性
            # clip_text_features = self.linear_1024_576(
            #     self.clip_text_features.float().to(support_cam.device))  # torch.Size([20, 625])
            clip_text_features = self.linear_1024_576(
                self.clip_text_features.float().to(support_mask.device))  # torch.Size([20, 625])
            clip_text_features = clip_text_features[class_id]
            # 修改clip特征的张量
            batch, channel = clip_text_features.size()[:2]
            clip_text_features = clip_text_features.unsqueeze(2)
            clip_text_features = clip_text_features.unsqueeze(3)
            clip_text_features = clip_text_features.expand(batch, channel, 48, 48)


        _, channels, hm, wm = mix.size()


        corr = torch.cat((mix, clip_text_features), dim=1)   # 合并原型和预测掩码图
        corr1 = self.init_merge1(corr)
        corr_mix = self.ASPP_meta1(corr1)
        # corr_mix = mix

        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(
                dim=0).values
        mix = torch.cat((corr_mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)
        mix = self.init_merge2(mix)
        mix = self.beta_conv2(mix)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(
                dim=0).values

        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)
        mix = self.init_merge3(mix)
        mix = self.beta_conv3(mix)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask

    def mask_feature(self, features, support_mask):
        mask = F.interpolate(support_mask.unsqueeze(1).float(), features.size()[2:], mode='bilinear', align_corners=True)
        features = features * mask
        return features

    def mask_feature2(self, features, support_mask):
        nshot = support_mask.size(1)
        masks = [support_mask[0, i] for i in range(nshot)]
        if len(features) != nshot:
            raise ValueError(f"特征数量({len(features)})与掩码数量({nshot})不匹配")

        masked_features = []
        for n in range(nshot):  # 遍历每个样本
            shot_features = []
            for idx, feat in enumerate(features[n]):
                # 跳过不需要处理的尺度
                if feat is None or idx < self.stack_ids[0]:
                    shot_features.append(feat)
                    continue

                # 调整当前样本的掩码大小
                mask_resized = F.interpolate(
                    masks[n].unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
                    size=feat.size()[2:],
                    mode='bilinear',
                    align_corners=True
                )

                # 应用掩码并保留梯度
                masked_feat = feat * mask_resized
                shot_features.append(masked_feat)

            masked_features.append(shot_features)
        return masked_features

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            # 确定当前卷积层的输入通道数
            # 如果是第一个卷积层，输入通道数为 in_channel
            # 否则，输入通道数为上一个卷积层的输出通道数
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            # 计算卷积层的填充大小，通常为卷积核大小的一半
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)

    def CMGM(self, query_feat_high, supp_feat_high, supp_feat_high_fg, s_y, fts_size, nshot):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        corr_query_mask_fg_list = []
        cosine_eps = 1e-7
        for st in range(nshot):
            # tmp_mask = (s_y[:, st, :, :] == 1).float().unsqueeze(1)
            # tmp_mask = F.interpolate(tmp_mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)

            tmp_supp_feat = supp_feat_high[:, st, ...]
            tmp_supp_fg = supp_feat_high_fg[:, st, ...]
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s_fg = self.high_avg_pool(tmp_supp_fg.flatten(2).transpose(-2, -1))
            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            tmp_supp_fg = s_fg
            tmp_supp_fg = tmp_supp_fg.contiguous()
            tmp_supp_fg = tmp_supp_fg.contiguous()
            tmp_supp_norm_fg = torch.norm(tmp_supp_fg, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)

            similarity_fg = torch.bmm(tmp_supp_fg, tmp_query) / (torch.bmm(tmp_supp_norm_fg, tmp_query_norm) + cosine_eps)
            similarity_fg = similarity_fg.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity_fg = (similarity_fg - similarity_fg.min(1)[0].unsqueeze(1)) / (similarity_fg.max(1)[0].unsqueeze(1) - similarity_fg.min(1)[0].unsqueeze(1) + cosine_eps)

            # similarity = torch.nn.functional.relu(similarity - similarity_bg * 0.5)

            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)

            corr_query_fg = similarity_fg.view(bsize, 1, sp_sz, sp_sz)
            corr_query_fg = F.interpolate(corr_query_fg, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_fg_list.append(corr_query_fg)

        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask_fg = torch.cat(corr_query_mask_fg_list, 1).mean(1).unsqueeze(1)
        corr_query_mix = torch.cat((corr_query_mask, corr_query_mask_fg), dim=1)
        return corr_query_mix

    def CMGM1(self, query_feat_high, supp_feat_high, fts_size, nshot):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        similaritys = []
        cosine_eps = 1e-7
        for st in range(nshot):

            tmp_supp_feat = supp_feat_high[:, st, ...]
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)


            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)

        #     corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
        #     corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
        #     corr_query_mask_list.append(corr_query)
        #
        # corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        return similarity
    def qinhe(self, query_feat_high, supp_feat_high):
        # 移除high_avg_pool，保留空间信息
        q = query_feat_high.flatten(2)  # [bs, c, h*w]
        s = supp_feat_high.flatten(2)  # [bs, c, h*w]

        # 转置为 [bs, h*w, c]
        q = q.permute(0, 2, 1).contiguous()  # [bs, h*w, c]
        s = s.permute(0, 2, 1).contiguous()  # [bs, h*w, c]

        # 统一计算范数：沿最后一个维度（c）
        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)  # [bs, h*w, 1]
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)  # [bs, h*w, 1]

        # 点积 + 归一化
        cosine_eps = 1e-7
        dot_product = torch.bmm(q, s.transpose(1, 2))  # [bs, h*w, h*w]
        norms_product = torch.bmm(q_norm, s_norm.transpose(1, 2))  # [bs, h*w, h*w]
        similarity = dot_product / (norms_product + cosine_eps)  # [bs, h*w, h*w]
        return similarity

    def qinhe1(self, query_feat_high, supp_feat_high):
        bsize, c, sp_sz, _ = query_feat_high.size()[:]
        cosine_eps = 1e-7
        q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
        s = self.high_avg_pool(supp_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        return similarity
