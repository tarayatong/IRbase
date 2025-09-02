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
import torchvision.transforms  as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from thop import profile

from segment_anything_training.build_IRSAM import build_sam_IRSAM

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, get_im_gt_name_list
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.metric import PD_FA, ROCMetric
from utils.loss_mask import DICE_loss
from utils.log import initialize_logger
import utils.misc as misc

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model_type", type=str, default="vit_l",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True,
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
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=10, type=int)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(valid_datasets, args):
    # --- Step 1: Valid dataset ---
    print("--- create train dataloader ---")
    train_im_gt_list = get_im_gt_name_list(valid_datasets, flag="train")
    train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_train,
                                                           training=True)
    print(len(train_dataloaders), " valid dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_list(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.dataloader_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 2: Load pretrained Network---
    net = build_sam_IRSAM(checkpoint=args.checkpoint)
    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    criterion = DICE_loss  # Assuming you use DICE_loss for segmentation tasks
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

        # Prepare to store the results
        training_loss_list = []
        evaluation_metrics_list = []

        # Loop for training and evaluating for 20 epochs
        for epoch in range(1, 21):  # 20 epochs
            print(f"--- Epoch {epoch} ---")
            # Training step
            train_metrics = train(net, train_dataloaders, optimizer, criterion)
            training_loss_list.append(train_metrics['loss'])

            # Evaluation step after each epoch
            print(f"Evaluating after epoch {epoch}...")
            eval_metrics = evaluate(net, valid_dataloaders)

            # Store evaluation metrics
            evaluation_metrics_list.append(eval_metrics)

            # Logging the results
            print(f"Epoch {epoch} results: IoU={eval_metrics['iou']}, nIoU={eval_metrics['niou']}, PD={eval_metrics['pd']}, FA={eval_metrics['fa']}")
            print(f"Training loss for epoch {epoch}: {train_metrics['loss']}")

            # Save model checkpoint if necessary
            if epoch % args.model_save_fre == 0:
                checkpoint_path = os.path.join(args.output, f"checkpoint_epoch_{epoch}.pth")
                torch.save(net.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")

        # Optionally save loss and metrics history to a file
        np.savetxt(os.path.join(args.output, 'training_loss.txt'), training_loss_list, fmt='%.6f')
        np.savetxt(os.path.join(args.output, 'evaluation_metrics.txt'), evaluation_metrics_list, fmt='%.6f')

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

            # Create the batched input for the model
            batched_input = []
            for b_i in range(inputs_val.shape[0]):
                dict_input = dict()
                dict_input['image'] = inputs_val[b_i] # Single image in the batch
                dict_input['original_size'] = shapes_val[b_i]
                
                # Add optional inputs if they exist
                if point_coords is not None:
                    dict_input['point_coords'] = point_coords[b_i]  # Add point coordinates
                if point_labels is not None:
                    dict_input['point_labels'] = point_labels[b_i] # Add point labels
                if boxes is not None:
                    dict_input['boxes'] = boxes[b_i]  # Add bounding box
                if mask_inputs is not None:
                    dict_input['mask_inputs'] = mask_inputs[b_i]  # Add mask inputs

                batched_input.append(dict_input)

            masks, edges = net(batched_input)

            torch.cuda.synchronize()

            IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
            Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

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
    Pd_Fa = PD_FA(1, 10)

    IoU_metric.reset()
    nIoU_metric.reset()
    Pd_Fa.reset()

    epoch_loss = 0  # To track the loss for this epoch
    tbar = tqdm(train_dataloaders)
    for data_train in tbar:
        # Assuming data_val is now a dictionary with the required fields.
        inputs_val = data_train['image']  # Tensor with shape [B, 3, H, W]
        labels_ori = data_train['label']  # Ground truth labels, shape [B, H, W]
        shapes_val = data_train['shape']  # Image shapes (original sizes)
        
        # Additional fields (if present in the dataset)
        point_coords = data_train.get('point_coords', None)  # Optional point coordinates
        point_labels = data_train.get('point_labels', None)  # Optional point labels
        boxes = data_train.get('boxes', None)  # Optional bounding boxes
        mask_inputs = data_train.get('mask_inputs', None)  # Optional mask inputs

        # Move data to GPU if available
        if torch.cuda.is_available():
            inputs_val = inputs_val.cuda()
            labels_ori = labels_ori.cuda()

        # Create the batched input for the model
        batched_input = []
        for b_i in range(inputs_val.shape[0]):
            dict_input = dict()
            dict_input['image'] = inputs_val[b_i] # Single image in the batch
            dict_input['original_size'] = shapes_val[b_i]
            
            # Add optional inputs if they exist
            if point_coords is not None:
                dict_input['point_coords'] = point_coords[b_i]  # Add point coordinates
            if point_labels is not None:
                dict_input['point_labels'] = point_labels[b_i] # Add point labels
            if boxes is not None:
                dict_input['boxes'] = boxes[b_i]  # Add bounding box
            if mask_inputs is not None:
                dict_input['mask_inputs'] = mask_inputs[b_i]  # Add mask inputs

            batched_input.append(dict_input)

        # Forward pass
        optimizer.zero_grad()
        masks, edges = net(batched_input)

        # Compute loss (use your specific loss function here)
        loss, _ = criterion(masks, labels_ori)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update metrics
        IoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
        nIoU_metric.update(masks.cpu(), (labels_ori / 255.).cpu().detach())
        Pd_Fa.update(masks.cpu(), (labels_ori / 255.).cpu().detach())

        FA, PD = Pd_Fa.get(len(train_dataloaders))
        _, IoU = IoU_metric.get()
        _, nIoU = nIoU_metric.get()

        tbar.set_description('Loss:%.8lf, IoU:%f, nIoU:%f, PD:%.8lf, FA:%.8lf'
                        % (loss.item(), IoU, nIoU, PD[0], FA[0]))

    # Calculate average loss for the epoch
    epoch_loss /= len(train_dataloaders)

    # Get final metrics for this epoch
    _, IoU = IoU_metric.get()
    _, nIoU = nIoU_metric.get()
    FA, PD = Pd_Fa.get(len(train_dataloaders))

    metric['loss'] = epoch_loss
    metric['iou'] = IoU
    metric['niou'] = nIoU
    metric['pd'] = PD[0]
    metric['fa'] = FA[0]

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
