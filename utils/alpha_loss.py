"""
Alpha损失函数模块
用于优化基于alpha融合的mask预测
"""

import torch
import torch.nn.functional as F

def AlphaLoss(masks, bgs, alpha, edges, labels_ori):
    """
    masks, bgs, alpha, edges, labels_ori
    计算alpha相关损失
    优化目标: (y-p)dot(y-q)/||(y-q)||，计算L2损失
    """
    # 确保输入格式正确
    # p = masks (不需要激活，直接使用logits)
    # q = bgs (不需要激活，直接使用logits) 
    # y = labels_ori (需要归一化到0-1)
    p = masks  # [B, C, H, W] - 直接使用logits
    q = bgs    # [B, 1, H, W] - 直接使用logits
    y = labels_ori.repeat(1, 32, 1, 1)/255.  # [B, 1, H, W] 与q维度匹配

    # y_minus_p = y - p  # [B, C, H, W]
    # y_minus_q = y - q  # [B, 1, H, W]
    # dot_product = (y_minus_p * y_minus_q).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    # y_minus_q_norm = torch.norm(y_minus_q, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    # epsilon = 1e-8
    # y_minus_q_norm = torch.clamp(y_minus_q_norm, min=epsilon)
    # # 计算目标值: (y-p)dot(y-q)/||(y-q)||
    # target = dot_product / y_minus_q_norm  # [B, 1, H, W]
    
    target = ((p-y)*q).sum(dim=1, keepdim=True)/(torch.norm(q, p=2, dim=1, keepdim=True)+1e-8)

    # 计算alpha与目标值的L2损失
    alpha_loss_val = F.mse_loss(alpha, target)
    
    return alpha_loss_val

