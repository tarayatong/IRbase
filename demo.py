# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as ops
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from thop import profile

from segment_anything_training.build_IRSAM import build_sam_IRSAM

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, \
    get_im_gt_name_list
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.metric import PD_FA, ROCMetric
from utils.loss_mask import DICE_loss
from utils.log import initialize_logger
from utils.mask_cache import MaskCache, generate_masks_for_dataset
from utils.alpha_loss import AlphaLoss
import utils.misc as misc

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model_type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, #default='./workdirs/WPMD+multi+encoder_s4/best.pth',#required=True,
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--no_prompt_checkpoint", type=str, default=None,
                        help="The path to the SAM checkpoint trained with no prompt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=1001, type=int)
    parser.add_argument('--dataloader_size', default=[512, 512], type=list)
    parser.add_argument('--batch_size_train', default=2, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=10, type=int)
    parser.add_argument('--update_mask_cache', default=False, type=bool)
    parser.add_argument('--mask_cache_update_freq', default=1, type=int,
                        help="Frequency of mask cache updates. Update cache every k epochs. Default is 1 (every epoch).")
    parser.add_argument('--use_mask_cache', default=False, type=bool,
                        help="Whether to use mask cache during training. If False, disables all mask caching functionality.")

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()

def check_checkpoint_compatibility(net, checkpoint_path):
    """
    检查checkpoint是否与当前网络结构兼容
    
    Args:
        net: 当前的网络模型
        checkpoint_path: checkpoint文件路径
        
    Returns:
        bool: True如果兼容，False如果不兼容
    """
    try:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False
            
        # 加载checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 获取当前网络的state_dict
        current_state_dict = net.state_dict()
        
        # 检查关键参数的形状是否匹配
        checkpoint_keys = set(checkpoint.keys())
        current_keys = set(current_state_dict.keys())
        
        # 检查缺失的键
        missing_keys = current_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - current_keys
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        
        # 检查共同键的形状是否匹配
        shape_mismatch = False
        for key in checkpoint_keys & current_keys:
            if checkpoint[key].shape != current_state_dict[key].shape:
                print(f"Shape mismatch for {key}: checkpoint {checkpoint[key].shape} vs current {current_state_dict[key].shape}")
                shape_mismatch = True
        
        # 如果有形状不匹配或者关键键缺失，认为不兼容
        if shape_mismatch or missing_keys:
            return False
        
        print("Checkpoint compatibility check passed")
        return True
        
    except Exception as e:
        print(f"Error checking checkpoint compatibility: {e}")
        return False

def evaluate_save_masks(valid_datasets, args):
    valid_im_gt_list = get_im_gt_name_list(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    net = build_sam_IRSAM(checkpoint=args.checkpoint)
    if torch.cuda.is_available():
        net.cuda()
    image_paths, predicted_masks = generate_masks_for_dataset(net, valid_dataloaders[0])
    output_path = os.path.join(args.output, "predicted_masks")
    os.makedirs(output_path, exist_ok=True)
    for i, img_path in enumerate(image_paths):
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), predicted_masks[i].cpu().numpy() * 255)




def main(valid_datasets, args):
    # --- Step 1: Initialize mask cache (if enabled) ---
    mask_cache = None
    if args.use_mask_cache:
        mask_cache_dir = os.path.join(args.output, "mask_cache")
        mask_cache = MaskCache(mask_cache_dir, dataset_name=valid_datasets[0]["name"])
        print(f"Mask cache initialized: {mask_cache.get_cache_info()}")
    else:
        print("Mask cache is disabled by --use_mask_cache=False")
    
    # --- Step 2: Valid dataset ---
    print("--- create train dataloader ---")
    train_im_gt_list = get_im_gt_name_list(valid_datasets, flag="train")
    print(len(train_im_gt_list), " training images found")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_list(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 3: Load pretrained Network---
    net = build_sam_IRSAM(checkpoint=args.checkpoint, use_mask_cache=args.use_mask_cache)  # 传递use_mask_cache参数
    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    criterion = DICE_loss  # Assuming you use DICE_loss for segmentation tasks
    
    os.makedirs(args.output, exist_ok=True)
    
    # --- Step 4: Handle initial mask cache logic ---
    initial_cache_available = False
    
    if args.use_mask_cache:
        print("--- Checking initial mask cache status ---")
        
        if mask_cache.has_cache_for_epoch(0):
            print("Initial mask cache already exists, will use it directly")
            initial_cache_available = True
        else:
            print("No initial mask cache found")
            # 检查是否有checkpoint可以用来生成初始cache
            if args.checkpoint and os.path.exists(args.checkpoint):
                print(f"Checking checkpoint compatibility: {args.checkpoint}")
                if check_checkpoint_compatibility(net, args.checkpoint):
                    print("Checkpoint is compatible, generating initial mask cache...")
                    try:
                        # 创建用于生成初始mask的数据加载器
                        initial_mask_dataloaders, _ = create_dataloaders(train_im_gt_list,
                                                                       my_transforms=[
                                                                           Resize(args.dataloader_size)
                                                                       ],
                                                                       batch_size=args.batch_size_valid,
                                                                       training=True,
                                                                       mask_cache=None)  # 不使用cache
                        
                        # 使用checkpoint生成初始mask预测结果
                        image_paths, predicted_masks = generate_masks_for_dataset(net, initial_mask_dataloaders)
                        
                        # 保存到缓存（epoch 0表示初始checkpoint生成的mask）
                        mask_cache.save_epoch_masks(0, image_paths, predicted_masks)
                        print(f"Generated and saved {len(image_paths)} initial masks from checkpoint")
                        initial_cache_available = True
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate initial mask cache: {e}")
                        initial_cache_available = False
                else:
                    print("Checkpoint is not compatible with current network structure")
                    print("Will proceed with training without initial mask cache")
                    initial_cache_available = False
            else:
                print("No checkpoint provided or checkpoint file not found")
                print("Will proceed with training without initial mask cache")
                initial_cache_available = False
    else:
        print("Mask cache is disabled, skipping cache initialization")
    # --- Step 3: Train or Evaluate ---
    if args.eval:
        if args.restore_model:
            print("Restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
            else:
                net.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        # Evaluation step only
        evaluate(net, valid_dataloaders)

    else:
        # If restore model is specified, load it
        if args.restore_model:
            print("Restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
            else:
                net.load_state_dict(torch.load(args.restore_model, map_location="cpu"))
        best_iou = 0
        # Loop for training and evaluating for 20 epochs
        for epoch in range(1, 201):  # 20 epochs
            print(f"--- Epoch {epoch} ---")
            if epoch == 1:
                # 第一轮训练：使用初始cache（如果可用且启用）
                if args.use_mask_cache and initial_cache_available:
                    print(f"Using available initial mask cache as mask_inputs for epoch 1")
                    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                           my_transforms=[
                                                                               Resize(args.dataloader_size)
                                                                           ],
                                                                           batch_size=args.batch_size_train,
                                                                           training=True,
                                                                           mask_cache=mask_cache)
                else:
                    if not args.use_mask_cache:
                        print("Mask cache disabled, training epoch 1 without mask_inputs")
                    else:
                        print("No initial mask cache available, training epoch 1 without mask_inputs")
                    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                           my_transforms=[
                                                                               Resize(args.dataloader_size)
                                                                           ],
                                                                           batch_size=args.batch_size_train,
                                                                           training=True,
                                                                           mask_cache=None)
            else:
                # 后续轮次使用最近一次更新的mask_cache（如果启用）
                if args.use_mask_cache:
                    # 找到最近一次更新的cache epoch
                    latest_cache_epoch = None
                    for check_epoch in range(epoch - 1, -1, -1):  # 从epoch-1往前查找
                        if mask_cache.has_cache_for_epoch(check_epoch):
                            latest_cache_epoch = check_epoch
                            break
                    
                    if latest_cache_epoch is not None:
                        print(f"Using cached masks from epoch {latest_cache_epoch} as mask_inputs")
                        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                               my_transforms=[
                                                                                   Resize(args.dataloader_size)
                                                                               ],
                                                                               batch_size=args.batch_size_train,
                                                                               training=True,
                                                                               mask_cache=mask_cache)
                    else:
                        print(f"No cached masks found for any previous epoch, training without mask_inputs")
                        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                               my_transforms=[
                                                                                   Resize(args.dataloader_size)
                                                                               ],
                                                                               batch_size=args.batch_size_train,
                                                                               training=True,
                                                                               mask_cache=None)
                else:
                    print(f"Mask cache disabled, training epoch {epoch} without mask_inputs")
                    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                                           my_transforms=[
                                                                               Resize(args.dataloader_size)
                                                                           ],
                                                                           batch_size=args.batch_size_train,
                                                                           training=True,
                                                                           mask_cache=None)
            
            # Training step
            train_metrics = train(net, train_dataloaders, optimizer, criterion)

            # Check if we should update mask cache based on frequency
            should_update_cache = (args.use_mask_cache and args.update_mask_cache and 
                                 epoch % args.mask_cache_update_freq == 0)
            
            if should_update_cache:
                # 训练完成后，生成这个epoch的mask预测结果并保存
                print(f"Generating masks for epoch {epoch} (update frequency: every {args.mask_cache_update_freq} epochs)...")
                try:
                    # 创建一个单独的数据加载器用于生成mask（不使用cache，避免循环依赖）
                    mask_gen_dataloaders, _ = create_dataloaders(train_im_gt_list,
                                                            my_transforms=[
                                                                Resize(args.dataloader_size)
                                                            ],
                                                            batch_size=args.batch_size_valid,  # 使用较小的batch size
                                                            training=True,  # 使用training=True来得到单个dataloader
                                                            mask_cache=None)  # 不使用cache
                    
                    # 生成mask预测结果
                    image_paths, predicted_masks = generate_masks_for_dataset(net, mask_gen_dataloaders)
                    
                    # 保存到缓存
                    mask_cache.save_epoch_masks(epoch, image_paths, predicted_masks)
                    print(f"Saved {len(image_paths)} masks for epoch {epoch}")
                
                except Exception as e:
                    print(f"Warning: Failed to generate/save masks for epoch {epoch}: {e}")
            elif not args.use_mask_cache:
                print(f"Mask cache disabled, skipping mask generation for epoch {epoch}")
            elif not args.update_mask_cache:
                print(f"Mask cache update disabled, skipping mask generation for epoch {epoch}")
            elif args.use_mask_cache and args.update_mask_cache:
                print(f"Skipping mask cache update for epoch {epoch} (frequency: every {args.mask_cache_update_freq} epochs, next update: epoch {((epoch // args.mask_cache_update_freq) + 1) * args.mask_cache_update_freq})")

            # Evaluation step after each epoch
            print(f"Evaluating after epoch {epoch}...")
            eval_metrics = evaluate(net, valid_dataloaders)

            # Logging the results
            print(
                f"Epoch {epoch} results: IoU={eval_metrics['iou']}, nIoU={eval_metrics['niou']}, PD={eval_metrics['pd']}, FA={eval_metrics['fa']}")
            print(f"Training loss for epoch {epoch}: {train_metrics['loss']}")

            # Save model checkpoint if necessary
            if eval_metrics['iou'] > best_iou:
                best_iou = eval_metrics['iou']
                checkpoint_path = os.path.join(args.output, f"best.pth")
                torch.save(net.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")

            output_path = os.path.join(args.output, 'evaluation_metrics.txt')
            with open(output_path, 'a') as f:
                line = f"epoch: {epoch}, loss: {train_metrics['loss']}, "
                line += ", ".join([f"{k}: {v:.6f}" for k, v in eval_metrics.items()])
                f.write(line + "\n")

        print("Training complete!")


def evaluate(net, valid_dataloaders):
    net.eval()
    metric = dict()

    IoU_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

    ROC = ROCMetric(1, 10)
    Pd_Fa = PD_FA(1, 10)

    IoU_metric.reset()
    nIoU_metric.reset()
    Pd_Fa.reset()
    for k in range(len(valid_dataloaders)):
        valid_dataloader = valid_dataloaders[k]

        tbar = tqdm(valid_dataloader)
        for data_val in tbar:
            # Assuming data_val is now a dictionary with the required fields.
            inputs_val = data_val['image']  # Tensor with shape [B, 3, H, W]
            labels_ori = data_val['label']  # Ground truth labels, shape [B, H, W]
            shapes_val = data_val['shape']  # Image shapes (original sizes)

            # Additional fields (if present in the dataset)
            point_coords = data_val.get('point_coords', None)  # Optional point coordinates
            point_labels = data_val.get('point_labels', None)  # Optional point labels
            boxes = data_val.get('boxes', None)  # Optional bounding boxes
            mask_inputs = data_val.get('mask_inputs', None)  # Optional mask inputs

            # Move data to GPU if available
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_ori = labels_ori.cuda()
                # mask_inputs = mask_inputs.cuda()

            # Create the batched input for the model
            batched_input = []
            for b_i in range(inputs_val.shape[0]):
                dict_input = dict()
                dict_input['image'] = inputs_val[b_i]  # Single image in the batch
                dict_input['original_size'] = shapes_val[b_i]

                # Add optional inputs if they exist
                if point_coords is not None:
                    dict_input['point_coords'] = point_coords[b_i]  # Add point coordinates
                if point_labels is not None:
                    dict_input['point_labels'] = point_labels[b_i]  # Add point labels
                if boxes is not None:
                    dict_input['boxes'] = boxes[b_i]  # Add bounding box
                if mask_inputs is not None:
                    dict_input['mask_inputs'] = mask_inputs[b_i]  # Add mask inputs

                batched_input.append(dict_input)

            outputs, masks, edges, _ = net(batched_input)

            torch.cuda.synchronize()

            # 使用outputs作为最终预测结果进行评估
            IoU_metric.update(outputs.cpu(), (labels_ori / 255.).cpu().detach())
            nIoU_metric.update(outputs.cpu(), (labels_ori / 255.).cpu().detach())
            Pd_Fa.update(outputs.cpu(), (labels_ori / 255.).cpu().detach())

            FA, PD = Pd_Fa.get(len(valid_dataloader))
            _, IoU = IoU_metric.get()
            _, nIoU = nIoU_metric.get()

            tbar.set_description('IoU:%f, nIoU:%f, PD:%.8lf, FA:%.8lf'
                                 % (IoU, nIoU, PD[0], FA[0]))

        metric['iou'] = IoU
        metric['niou'] = nIoU
        metric['pd'] = PD[0]
        metric['fa'] = FA[0]
    return metric


def train(net, train_dataloaders, optimizer, criterion):
    net.train()
    metric = dict()

    IoU_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)

    ROC = ROCMetric(1, 10)
    # Pd_Fa = PD_FA(1, 10)

    IoU_metric.reset()
    nIoU_metric.reset()
    # Pd_Fa.reset()

    epoch_loss = 0  # To track the loss for this epoch
    tbar = tqdm(train_dataloaders)
    batch_count = 0
    for data_train in tbar:
        batch_count += 1
        
        # Assuming data_val is now a dictionary with the required fields.
        inputs_val = data_train['image']  # Tensor with shape [B, 3, H, W]
        labels_ori = data_train['label']  # Ground truth labels, shape [B, H, W]
        shapes_val = data_train['shape']  # Image shapes (original sizes)

        # Additional fields (if present in the dataset)
        point_coords = data_train.get('point_coords', None)  # Optional point coordinates
        point_labels = data_train.get('point_labels', None)  # Optional point labels
        boxes = data_train.get('boxes', None)  # Optional bounding boxes
        mask_inputs = data_train.get('mask_inputs', None)  # Optional mask inputs
        edges = data_train.get('edge', None)  # Optional edge inputs

        # Move data to GPU if available
        if torch.cuda.is_available():
            inputs_val = inputs_val.cuda()
            labels_ori = labels_ori.cuda()
            edges = edges.cuda()
            if mask_inputs is not None:
                mask_inputs = mask_inputs.cuda()

        # Create the batched input for the model
        batched_input = []
        for b_i in range(inputs_val.shape[0]):
            dict_input = dict()
            dict_input['image'] = inputs_val[b_i]  # Single image in the batch
            dict_input['original_size'] = shapes_val[b_i]

            # Add optional inputs if they exist
            if point_coords is not None:
                dict_input['point_coords'] = point_coords[b_i]  # Add point coordinates
            if point_labels is not None:
                dict_input['point_labels'] = point_labels[b_i]  # Add point labels
            if boxes is not None:
                dict_input['boxes'] = boxes[b_i]  # Add bounding box
            if mask_inputs is not None:
                # 检查是否是真正的cached mask（非全零张量）
                current_mask = mask_inputs[b_i]
                if torch.sum(current_mask) > 0:  # 如果不是全零张量
                    dict_input['mask_inputs'] = current_mask  # Add mask inputs，添加batch维度
            if edges is not None:
                dict_input['edge'] = edges[b_i]  # Add edge inputs

            batched_input.append(dict_input)

        # Forward pass
        optimizer.zero_grad()
        
<<<<<<< HEAD
        # 检查是否使用alpha融合
        if hasattr(net.mask_decoder, 'use_alpha') and net.mask_decoder.use_alpha:
            outputs, masks, bgs, alpha = net(batched_input)
            
            # 计算IoU损失
            iou_loss, _ = criterion(outputs, labels_ori/255.)
            edge_loss = F.binary_cross_entropy(bgs, edges/255.)
            alpha_loss= AlphaLoss(masks, bgs, alpha, edges, labels_ori)

            # 使用Alpha损失函数
            loss = iou_loss + 10*edge_loss + alpha_loss
        else:
            outputs, masks, bgs = net(batched_input)
            
            # 原始损失计算
            iou_loss, _ = criterion(outputs, labels_ori/255.)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(masks), labels_ori/255.)
            edge_loss = F.binary_cross_entropy(torch.sigmoid(bgs), edges/255.)
            loss = iou_loss + 10*bce_loss + 10*edge_loss
        
=======
        # 组合总损失
        loss = iou_loss + 10*edge_loss
<<<<<<< HEAD
>>>>>>> 6f2920dcac4f24ff417192bab7905c2d225fdb87
=======
>>>>>>> betterbase
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update metrics
        IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
        nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
        # Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

        # FA, PD = Pd_Fa.get(len(train_dataloaders))
        _, IoU = IoU_metric.get()
        _, nIoU = nIoU_metric.get()

        tbar.set_description('Loss:%.8lf, IoU:%f, nIoU:%f, '
                             % (loss.item(), IoU, nIoU))  # , PD:%.8lf, FA:%.8lf, PD[0], FA[0]

    # Calculate average loss for the epoch
    epoch_loss /= len(train_dataloaders)

    # Get final metrics for this epoch
    _, IoU = IoU_metric.get()
    _, nIoU = nIoU_metric.get()
    # FA, PD = Pd_Fa.get(len(train_dataloaders))

    metric['loss'] = epoch_loss
    metric['iou'] = IoU
    metric['niou'] = nIoU
    # metric['pd'] = PD[0]
    # metric['fa'] = FA[0]

    return metric


if __name__ == "__main__":
    # --------------- Configuring the Valid datasets ---------------
    dataset_val_nuaa = {"name": "Sirstv2_512",
                        "im_dir": "datasets/Sirstv2_512/test_images",
                        "gt_dir": "datasets/Sirstv2_512/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_NUDT = {"name": "NUDT",
                        "im_dir": "datasets/NUDT-SIRST00/test_images",
                        "gt_dir": "datasets/NUDT-SIRST00/test_masks",
                        "im_ext": ".png",
                        "gt_ext": ".png"}

    dataset_val_IRSTD = {"name": "IRSTD",
                         "im_dir": "datasets/IRSTD-1k/images",
                         "gt_dir": "datasets/IRSTD-1k/masks",
                         "im_ext": ".png",
                         "gt_ext": ".png"}

    valid_datasets = [dataset_val_IRSTD]

    args = get_args_parser()

    main(valid_datasets, args)
    # evaluate_save_masks(valid_datasets, args)

