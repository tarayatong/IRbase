# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from ..utils.dysample import DySample


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 0,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            mask_cache: bool = False,  # 新增参数
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
          mask_cache (bool): whether to enable mask cache functionality
            for dense_prompt_embeddings fusion
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.mask_cache = mask_cache  # 新增属性

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 使用DySample+Conv替代ConvTranspose2d，用Sequential包装
        self.output_upscaling = nn.Sequential(
            DySample(transformer_dim, scale=2),
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            DySample(transformer_dim // 4, scale=2),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1),
            activation(),
        )
        # 直接卷积头生成 masks/bg（不再使用tokens超网络）
        self.mask_head = nn.Conv2d(transformer_dim // 8, self.num_mask_tokens - 1, kernel_size=1)
        self.bg_head = nn.Conv2d(transformer_dim // 8, 1, kernel_size=1)

        # edge tokens
        self.edge_token = nn.Embedding(1, transformer_dim)
        self.edge_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # 使用DySample+Conv替代ConvTranspose2d，用Sequential包装
        self.compress_vit_feat = nn.Sequential(
            DySample(256, scale=2),
            nn.Conv2d(256, transformer_dim, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            DySample(transformer_dim, scale=2),
            nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=3, padding=1)
        )
        # 使用DySample+Conv替代ConvTranspose2d，用Sequential包装
        self.embedding_encoder = nn.Sequential(
            DySample(transformer_dim, scale=2),
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            DySample(transformer_dim // 4, scale=2),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1)
        )
        # 这里的ConvTranspose2d参数(3,1,1)实际上是普通卷积，直接用Conv2d替换
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

        # 简单的仅基于encoder输出的自注意力编码器（3层）
        self.attn_dim = transformer_dim
        self.attn_heads = 8
        self.attn_layers = nn.ModuleList([
            SelfAttentionBlock(self.attn_dim, self.attn_heads) for _ in range(3)
        ])

        # 无tokens分支：用全局上下文生成用于超网络的token表示
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.to_iou_token = nn.Linear(self.attn_dim, self.attn_dim)
        # 生成 num_mask_tokens 个token（包含edge token在内）
        self.to_mask_tokens = nn.Linear(self.attn_dim, self.attn_dim * (self.num_mask_tokens))

    def forward(
            self,
            image_embeddings: torch.Tensor,
            edge_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted outputs (for IoU loss)
          torch.Tensor: batched predicted masks (for BCE loss)
          torch.Tensor: batched predicted background/edges (for edge BCE loss)
        """
        # edge_features = edge_embeddings.permute(0, 3, 1, 2)
        edge_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(edge_embeddings)  # qian+shen
        # edge_features = self.compress_vit_feat(edge_embeddings)  # final
        # edge_features = self.embedding_encoder(image_embeddings)  # shen

        outputs, masks, bg = self.predict_masks(
            image_embeddings=image_embeddings,
            edge_embeddings=edge_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        outputs = outputs[:, mask_slice, :, :]
        masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return outputs, masks, bg

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            edge_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details.
        
        Returns:
          torch.Tensor: outputs (for IoU loss)
          torch.Tensor: masks (for BCE loss) 
          torch.Tensor: bg (for edge BCE loss)
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.edge_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if self.training and self.mask_cache:
            src = src * (1+torch.sigmoid(dense_prompt_embeddings))
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 自注意力仅处理encoder特征，不使用tokens
        # 输入 src: [B, C, H, W]
        for blk in self.attn_layers:
            src = blk(src)

        # 基于全局上下文生成 iou_token 与 mask_tokens
        # global_context: [B, C]
        global_context = self.global_pool(src).flatten(1)
        iou_token_out = self.to_iou_token(global_context)  # [B, C]
        mask_tokens_flat = self.to_mask_tokens(global_context)  # [B, C * num_mask_tokens]
        mask_tokens_out = mask_tokens_flat.view(b, self.num_mask_tokens, self.attn_dim)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        edge_embedding = self.embedding_maskfeature(upscaled_embedding) + edge_embeddings.repeat(b, 1, 1, 1) #

        # 直接通过卷积头得到 masks 和 bg
        masks = self.mask_head(upscaled_embedding)
        bg = self.bg_head(edge_embedding)

        # alpha = self.sigmoid(masks)

        # masks = masks*torch.sigmoid(masks - bg)
        outputs = masks-0.5*bg

        return outputs, masks, bg


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # -> [B, H*W, C]
        feat = x.flatten(2).transpose(1, 2)
        feat = self.norm1(feat)
        attn_out, _ = self.attn(feat, feat, feat, need_weights=False)
        feat = feat + attn_out
        feat = self.norm2(feat)
        feat = feat + self.mlp(feat)
        # -> [B, C, H, W]
        feat = feat.transpose(1, 2).view(B, C, H, W)
        return feat
