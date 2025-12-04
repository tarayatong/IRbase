"""
Alpha损失函数模块
用于优化基于alpha融合的mask预测
"""

import torch
import torch.nn.functional as F

def AlphaLoss(masks, bgs, alpha, edges, labels_ori, mode='geo'):
    masks = masks.detach()
    bgs = bgs.detach()
    y = labels_ori.repeat(1, 32, 1, 1) / 255.  # [B, 32, H, W] 与p维度匹配

    if mode == 'cos':
        # cos distance
        p_ = alpha #(1 + alpha) * masks - alpha * bgs
        cos_sim = F.cosine_similarity(p_, y, dim=1)
        alpha_loss_val = (1-cos_sim).mean()
    else:
        p_min = masks.min(dim=1, keepdim=True)[0]  # [B, 1, H, W] - 每个像素的最小值
        p_max = masks.max(dim=1, keepdim=True)[0]  # [B, 1, H, W] - 每个像素的最大值
        p = (masks - p_min) / (p_max - p_min + 1e-8)  # [B, C, H, W] - 每个像素向量归一化到[0,1]
        q_min = bgs.min(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        q_max = bgs.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        q = (bgs - q_min) / (q_max - q_min + 1e-8)  # [B, C, H, W] - 每个像素向量归一化到[0,1]
        p__min = alpha.min(dim=1, keepdim=True)[0]
        p__max = alpha.max(dim=1, keepdim=True)[0]
        p_ = (alpha - p__min) / (p__max - p__min + 1e-8)
        if mode == 'geo':
            # orthogonality condition
            target1 = ((y-p_) * (p-q)).sum(dim=1, keepdim=True)
            # target2 = (((1+alpha)*p-alpha*q)*(p-q)).sum(dim=1, keepdim=True)
            # target2 = ((p_) * (p-q)).sum(dim=1, keepdim=True)
            alpha_loss_val = F.mse_loss(target1, torch.zeros(target1.shape).cuda())
        else:
            # absolute distance
            # p_ = (1 + alpha) * p - alpha * q
            alpha_loss_val = F.mse_loss(y, p_)

    return alpha_loss_val

