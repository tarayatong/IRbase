"""
Alpha损失函数模块
用于优化基于alpha融合的mask预测
"""

import torch
import torch.nn.functional as F

def AlphaLoss(masks, bgs, alpha, edges, labels_ori):
    masks = masks.detach()
    bgs = bgs.detach()
    # masks: [B, C, H, W]，对每个(b,h,w)位置的C维向量单独归一化
    p_min = masks.min(dim=1, keepdim=True)[0]  # [B, 1, H, W] - 每个像素的最小值
    p_max = masks.max(dim=1, keepdim=True)[0]  # [B, 1, H, W] - 每个像素的最大值
    p = (masks - p_min) / (p_max - p_min + 1e-8)  # [B, C, H, W] - 每个像素向量归一化到[0,1]

    # bgs同样处理
    q_min = bgs.min(dim=1, keepdim=True)[0]  # [B, 1, H, W]
    q_max = bgs.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
    q = (bgs - q_min) / (q_max - q_min + 1e-8)  # [B, C, H, W] - 每个像素向量归一化到[0,1]

    p__min = alpha.min(dim=1, keepdim=True)[0]
    p__max = alpha.max(dim=1, keepdim=True)[0]
    p_ = (alpha - p__min) / (p__max - p__min + 1e-8)

    y = labels_ori.repeat(1, 32, 1, 1)/255.  # [B, 32, H, W] 与p维度匹配

    # y_minus_p = y - p  # [B, C, H, W]
    # y_minus_q = y - q  # [B, 1, H, W]
    # dot_product = (y_minus_p * y_minus_q).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    # y_minus_q_norm = torch.norm(y_minus_q, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    # epsilon = 1e-8
    # y_minus_q_norm = torch.clamp(y_minus_q_norm, min=epsilon)
    # # 计算目标值: (y-p)dot(y-q)/||(y-q)||
    # target = dot_product / y_minus_q_norm  # [B, 1, H, W]

    # orthogonality condition
    target1 = (y*(p-q)).sum(dim=1, keepdim=True)
    target2 = (p_*(p-q)).sum(dim=1, keepdim=True)
    alpha_loss_val = F.mse_loss(target1, target2)

    # absolute distance
    # p_ = (1 + alpha) * p - alpha * q
    # alpha_loss_val = F.mse_loss(y, p_)

    # cos distance
    # p_ = (1 + alpha) * p - alpha * q
    # cos_sim = F.cosine_similarity(p_, y, dim=1)
    # alpha_loss_val = (1-cos_sim).mean()

    return alpha_loss_val

