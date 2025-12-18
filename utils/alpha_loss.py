"""
Alpha损失函数模块
用于优化基于alpha融合的mask预测
"""

import torch
import torch.nn.functional as F


def compute_inverse_embedding(hyper_in_tokens, target_masks):
    """
    逆向计算 embedding：给定目标 mask，求解使得 token @ embedding = mask 的 embedding
    
    正向操作: masks = hyper_in @ img_embedding
    逆向操作: img_embedding = hyper_in⁺ @ masks (求解)
    
    Args:
        hyper_in_tokens: [b, num_tokens, c] - token向量，例如 (b, 1, 32)
        target_masks: [b, num_tokens, h, w] - 目标mask，例如 (b, 1, h, w)
    
    Returns:
        unknown_embedding: [b, c, h, w] - 计算出的embedding，例如 (b, 32, h, w)
    """
    b, num_tokens, c = hyper_in_tokens.shape  # b, 1, 32
    _, _, h, w = target_masks.shape  # b, 1, h, w
    
    # 将 target_masks reshape 成 [b, num_tokens, h*w]
    target_flat = target_masks.view(b, num_tokens, h * w)  # [b, 1, h*w]
    
    # 计算伪逆: [b, num_tokens, c] -> [b, c, num_tokens]
    # 例如: [b, 1, 32] -> [b, 32, 1]
    hyper_in_pinv = torch.linalg.pinv(hyper_in_tokens)
    
    # 矩阵乘法: [b, c, num_tokens] @ [b, num_tokens, h*w] = [b, c, h*w]
    # 例如: [b, 32, 1] @ [b, 1, h*w] = [b, 32, h*w]
    unknown_embedding_flat = torch.bmm(hyper_in_pinv, target_flat)
    
    # reshape 回 [b, c, h, w]
    # 例如: [b, 32, h*w] -> [b, 32, h, w]
    unknown_embedding = unknown_embedding_flat.view(b, c, h, w)
    
    return unknown_embedding


def AlphaLoss(out_dict, edges, labels_ori, mode='geo'):
    """
    计算 Alpha Loss
    
    Args:
        out_dict: decoder 返回的字典，包含:
            - img_embedding: [b, 32, h, w]
            - edge_embeddings: [b, 32, h, w]
            - alpha: [b, 32, h, w]
            - hyper_in: [b, 1, 32]
        edges: 边缘图 [b, 1, h, w]
        labels_ori: 标签 [b, 1, h, w]，值范围 [0, 255]
        mode: 'geo' (几何正交), 'cos' (余弦相似度), 'mse' (均方误差)
    """
    # 从字典中获取需要的张量
    masks = out_dict.get("img_embed")  # [b, 32, h, w]
    bgs = out_dict.get("edge_embed")  # [b, 32, h, w]
    alpha = out_dict.get("alpha")  # [b, 32, h, w] 或 None
    hyper_tokens = out_dict.get("hyper_tokens")  # [b, 1, 32]
    corrected_embed = out_dict.get("corrected_embed")
    
    # 检查必要的张量是否存在
    if masks is None or bgs is None or hyper_tokens is None:
        raise ValueError("out_dict 必须包含 'img_embedding'/'upscaled_embedding', 'edge_embeddings', 'hyper_in'")
    
    # detach 避免影响主要的梯度流
    # masks = masks.detach()
    # bgs = bgs.detach()
    # hyper_tokens = hyper_tokens.detach()
    # corrected_embed = corrected_embed.detach()
    
    # 将 labels 归一化到 [0, 1] 并保持形状 [b, 1, h, w]
    y = labels_ori / 255.0  # [B, 1, H, W]
    
    # 逆向计算理想的 embedding: 如果 hyper_tokens @ y_embedding = y，那么 y_embedding 是什么
    # y_embedding 的形状: [b, 32, h, w]
    y_embedding = compute_inverse_embedding(hyper_tokens, y)

    p_min = masks.min(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    p_max = masks.max(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    p = (masks - p_min) / (p_max - p_min + 1e-8)  # [b, 32, h, w]

    q_min = bgs.min(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    q_max = bgs.max(dim=1, keepdim=True)[0]  # [b, 1, h, w]
    q = (bgs - q_min) / (q_max - q_min + 1e-8)  # [b, 32, h, w]

    p__min = alpha.min(dim=1, keepdim=True)[0]
    p__max = alpha.max(dim=1, keepdim=True)[0]
    p_ = (alpha - p__min) / (p__max - p__min + 1e-8)  # [b, 32, h, w]

    y_min = y_embedding.min(dim=1, keepdim=True)[0]
    y_max = y_embedding.max(dim=1, keepdim=True)[0]
    y_ = (y_embedding - y_min) / (y_max - y_min + 1e-8)

    if mode == 'cos':
        # 余弦相似度模式：直接比较 alpha 和 y_embedding
        if alpha is None:
            raise ValueError("mode='cos' 需要 alpha，但 out_dict 中 alpha 为 None")
        cos_sim = F.cosine_similarity(p_, y_, dim=1)  # [b, h, w]
        alpha_loss_val = (1 - cos_sim).mean()
        
    elif mode == 'geo':
        # 几何正交模式：要求 (y_embedding - alpha) ⊥ (masks - bgs)
        if alpha is None:
            raise ValueError("mode='geo' 需要 alpha，但 out_dict 中 alpha 为 None")

        target1 = ((y_ - p_) * (p - q)).sum(dim=1, keepdim=True)  # [b, 1, h, w]
        alpha_loss_val = F.mse_loss(target1, torch.zeros_like(target1))
        
    else:
        if alpha is None:
            raise ValueError(f"mode='{mode}' 需要 alpha，但 out_dict 中 alpha 为 None")
        alpha_loss_val = F.mse_loss(p_, y_)

    return alpha_loss_val

