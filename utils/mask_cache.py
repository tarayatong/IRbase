# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Optional
import logging
from torch.nn import functional as F

class MaskCache:
    """
    管理训练过程中的mask缓存，用于在下一个epoch中提供上一轮训练的预测结果作为mask_input
    """
    
    def __init__(self, cache_dir: str, dataset_name: str = "default"):
        """
        初始化mask缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            dataset_name: 数据集名称，用于区分不同数据集的缓存
        """
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.current_epoch = 0
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存文件路径
        self.cache_file = os.path.join(cache_dir, f"mask_cache_{dataset_name}.pkl")
        
        # 内存中的缓存: {image_path: mask_tensor}
        self.mask_cache: Dict[str, torch.Tensor] = {}
        
        # 加载已有的缓存
        self._load_cache()
        
    def _load_cache(self):
        """从磁盘加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.mask_cache = cache_data.get('masks', {})
                    self.current_epoch = cache_data.get('epoch', 0)
                    logging.info(f"Loaded mask cache from {self.cache_file}, epoch: {self.current_epoch}")
            except Exception as e:
                logging.warning(f"Failed to load cache from {self.cache_file}: {e}")
                self.mask_cache = {}
                self.current_epoch = 0
        else:
            logging.info(f"No existing cache found at {self.cache_file}")
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            cache_data = {
                'masks': self.mask_cache,
                'epoch': self.current_epoch,
                'dataset_name': self.dataset_name
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.info(f"Saved mask cache to {self.cache_file}, epoch: {self.current_epoch}")
        except Exception as e:
            logging.error(f"Failed to save cache to {self.cache_file}: {e}")
    
    def save_epoch_masks(self, epoch: int, image_paths: List[str], masks: torch.Tensor):
        """
        保存一个epoch的所有mask预测结果
        
        Args:
            epoch: 当前epoch数
            image_paths: 图像路径列表
            masks: 预测的mask张量，形状为 [N, H, W] 或 [N, 1, H, W]
        """
        if len(image_paths) != masks.shape[0]:
            raise ValueError(f"Image paths count ({len(image_paths)}) != masks count ({masks.shape[0]})")
        
        # 确保mask是3维的 [N, H, W]
        if len(masks.shape) == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        
        # 更新缓存
        for i, img_path in enumerate(image_paths):
            # 标准化图像路径
            normalized_path = os.path.normpath(img_path)
            self.mask_cache[normalized_path] = masks[i].cpu().clone()
        
        self.current_epoch = epoch
        self._save_cache()
        
        logging.info(f"Saved {len(image_paths)} masks for epoch {epoch}")
    
    def get_mask(self, image_path: str) -> Optional[torch.Tensor]:
        """
        获取指定图像的缓存mask
        
        Args:
            image_path: 图像路径
            
        Returns:
            缓存的mask张量，如果不存在则返回None
        """
        normalized_path = os.path.normpath(image_path)
        return self.mask_cache.get(normalized_path, None)
    
    def has_cache_for_epoch(self, epoch: int) -> bool:
        """
        检查是否有指定epoch的缓存
        
        Args:
            epoch: epoch数
            
        Returns:
            如果有缓存返回True，否则返回False
        """
        return self.current_epoch >= epoch and len(self.mask_cache) > 0
    
    def clear_cache(self):
        """清空所有缓存"""
        self.mask_cache.clear()
        self.current_epoch = 0
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logging.info("Cleared all mask cache")
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        return {
            'cache_file': self.cache_file,
            'dataset_name': self.dataset_name,
            'current_epoch': self.current_epoch,
            'cached_images': len(self.mask_cache),
            'cache_exists': os.path.exists(self.cache_file)
        }


def generate_masks_for_dataset(net, dataloader, device='cuda'):
    """
    为整个数据集生成mask预测结果
    
    Args:
        net: 训练好的网络模型
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        image_paths: 图像路径列表
        masks: 预测的mask张量
    """
    net.eval()
    image_paths = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 获取图像路径
            if 'path' in data:
                batch_paths = data['path']
            elif 'ori_im_path' in data:
                batch_paths = data['ori_im_path']
            else:
                # 如果没有路径信息，使用index作为标识
                batch_paths = [f"batch_{batch_idx}_item_{i}" for i in range(data['image'].shape[0])]
            
            # 准备输入数据
            inputs_val = data['image']
            shapes_val = data['shape']
            
            if torch.cuda.is_available() and device == 'cuda':
                inputs_val = inputs_val.cuda()
            
            # 创建批量输入
            batched_input = []
            for b_i in range(inputs_val.shape[0]):
                dict_input = {
                    'image': inputs_val[b_i],
                    'original_size': shapes_val[b_i]
                }
                batched_input.append(dict_input)
            
            # 前向推理
            masks, _ = net(batched_input)
            
            # 收集结果
            image_paths.extend(batch_paths)
            all_masks.append(torch.sigmoid(masks).cpu())
    
    # 合并所有mask
    all_masks = torch.cat(all_masks, dim=0)
    
    return image_paths, all_masks
