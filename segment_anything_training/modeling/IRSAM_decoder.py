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
            use_alpha: bool = False,  # 是否使用余弦相似度alpha融合输出
            use_beta
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
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.num_mask_tokens = num_multimask_outputs+1
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
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=1),
            # activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.bg_head = nn.Sequential(
            nn.Conv2d(transformer_dim//8, transformer_dim//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim//8),
            nn.GELU(),
            nn.Conv2d(transformer_dim//8, 1, 1)
        )
        # 使用DySample+Conv替代ConvTranspose2d，用Sequential包装
        self.compress_vit_feat = nn.Sequential(
            DySample(160, scale=2),
            nn.Conv2d(160, transformer_dim, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            DySample(transformer_dim, scale=2),
            nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim//8),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=1),
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
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

        # 用卷积直接预测alpha，输入为 [masks, bg] 按通道拼接
        # 拼接后通道数为 (num_mask_channels + 1) = self.num_mask_tokens
        self.alpha_head = nn.Sequential(
            nn.Conv2d(transformer_dim//4, transformer_dim//8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim//8),
            nn.GELU(),
            nn.Conv2d(transformer_dim//8, transformer_dim//8, kernel_size=1, bias=False),
            # nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.gamma_head = nn.Sequential(
            nn.Conv2d(transformer_dim//4, transformer_dim//8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim//8),
            nn.GELU(),
            nn.Conv2d(transformer_dim//8, transformer_dim//8, kernel_size=1, bias=False),
            # nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        # self.alpha = nn.parameter(torch.ones(1,1,512,512))
        self.beta_head = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            edge_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_features = edge_embeddings.permute(0, 3, 1, 2)
        # edge_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(edge_features)  # qian+shen
        image_embeddings = self.embedding_maskfeature(image_embeddings)
        edge_features = self.compress_vit_feat(edge_features)  # qian
        # edge_features = self.embedding_encoder(image_embeddings)  # shen

        outputs, img_embed, edge_embed, bg, alpha = self.predict_masks(
            image_embeddings=image_embeddings,
            edge_embeddings=edge_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return outputs, img_embed, edge_embed, bg, alpha

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
        output_tokens = torch.cat([self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.output_upscaling(src)
        # edge_embedding = upscalesd_embedding + edge_embeddings 
        if self.use_alpha:
            alpha_in = torch.cat([upscaled_embedding, edge_embeddings], dim=1)
            alpha = self.alpha_head(alpha_in)
            gamma = self.gamma_head(alpha_in)
            img_embedding = gamma*upscaled_embedding - alpha*edge_embeddings
        else:
            img_embedding = upscaled_embedding

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](hs[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = img_embedding.shape
        masks = (hyper_in[:, :self.num_mask_tokens] @ img_embedding.view(b, c, h * w)).view(b, -1, h, w)
        bg = self.bg_head(edge_embeddings)

        if self.use_beta and self.use_alpha:
            beta = self.beta_head(torch.cat([masks, bg], dim=1))
            outputs = masks-beta*bg
            return outputs, upscaled_embedding, edge_embeddings, bg, alpha
        elif self.use_alpha:
            outputs = masks
            return outputs, upscaled_embedding, edge_embeddings, bg, img_embedding
        elif self.use_beta:
            beta = self.beta_head(torch.cat([masks, bg], dim=1))
            outputs = masks-beta*bg
            return outputs, None, None, bg, None
        else:
            outputs = masks
            return outputs, None, None, bg, None
        


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
