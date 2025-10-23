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
    if labels_ori.max() > 1.0:
        labels_ori = labels_ori / 255.0
    if edges.max() > 1.0:
        edges = edges / 255.0
    p = masks  # [B, C, H, W] - 直接使用logits
    q = edges    # [B, 1, H, W] - 直接使用logits
    y = labels_ori  # [B, 1, H, W] 与q维度匹配
    
    # 计算 (y-p)dot(y-q)/||(y-q)||
    # 首先计算 y-p 和 y-q
    y_minus_p = y - p  # [B, C, H, W]
    y_minus_q = y - q  # [B, 1, H, W]
    
    # 计算点积 (y-p)dot(y-q)
    # 需要将y_minus_p和y_minus_q在通道维度上对齐
    if y_minus_p.shape[1] != y_minus_q.shape[1]:
        # 如果通道数不匹配，将y_minus_q扩展到与y_minus_p相同的通道数
        y_minus_q = y_minus_q.expand_as(y_minus_p)
    
    dot_product = (y_minus_p * y_minus_q).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 计算 ||(y-q)|| 的L2范数
    y_minus_q_norm = torch.norm(y_minus_q, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    
    # 避免除零，添加小的epsilon
    epsilon = 1e-8
    y_minus_q_norm = torch.clamp(y_minus_q_norm, min=epsilon)
    
    # 计算目标值: (y-p)dot(y-q)/||(y-q)||
    target = dot_product / y_minus_q_norm  # [B, 1, H, W]
    
    # 计算alpha与目标值的L2损失
    alpha_loss_val = F.mse_loss(alpha, target)
    
    return alpha_loss_val

