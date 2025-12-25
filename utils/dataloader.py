# Copyright by HQ-SAM team
# All rights reserved.

# data loader
from __future__ import print_function, division

import cv2
import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T


# --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + '*' + datasets[i]["im_ext"])
        print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

        if datasets[i]["gt_dir"] == "":
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i][
                    "gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))

        name_im_gt_list.append({"dataset_name": datasets[i]["name"],
                                "im_path": tmp_im_list,
                                "gt_path": tmp_gt_list,
                                "im_ext": datasets[i]["im_ext"],
                                "gt_ext": datasets[i]["gt_ext"]})

    return name_im_gt_list


def get_im_gt_name_list(datasets, flag='train'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], " <<<---")
        
        # Initialize lists for image and ground truth paths
        tmp_im_list, tmp_gt_list = [], []

        # Read the filenames from the corresponding txt file
        if flag == 'train':
            list_txt = os.path.join(datasets[i]["txt_dir"], "train.txt")  #'datasets/IRSTD-1k/trainval.txt'
        else:
            list_txt = os.path.join(datasets[i]["txt_dir"], "test.txt")
        
        # Read the txt file containing filenames
        with open(list_txt, 'r') as f:
            filenames = f.readlines()

        # Construct the image paths from the filenames
        # if "NUDT" in datasets[i]["name"]:
        #     tmp_im_list = [datasets[i]["im_dir"] + os.sep + filename.strip() for filename in filenames]
        # else:
        tmp_im_list = [datasets[i]["im_dir"] + os.sep + filename.strip() + datasets[i]["im_ext"] for filename in
                       filenames]
        print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

        # Check if ground truth directory exists and construct the gt paths
        if datasets[i]["gt_dir"] == "":
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            if "v2" in datasets[i]["name"]:
                tmp_gt_list = [
                    datasets[i]["gt_dir"] + os.sep + filename.strip().split(os.sep)[-1].split(datasets[i]["im_ext"])[
                        0] + '_pixels0' + datasets[i]["gt_ext"]
                    for filename in filenames
                ]
            else:
                tmp_gt_list = [
                    datasets[i]["gt_dir"] + os.sep + filename.strip().split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i]["gt_ext"]
                    for filename in filenames
                ]
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))

        # Store image and ground truth paths in the list
        name_im_gt_list.append({
            "dataset_name": datasets[i]["name"],
            "im_path": tmp_im_list,
            "gt_path": tmp_gt_list,
            "im_ext": datasets[i]["im_ext"],
            "gt_ext": datasets[i]["gt_ext"]
        })

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False, mask_cache=None, img_size=512):
    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8
    my_transforms = []
    if training:
        my_transforms.append(RandomHFlip(prob=0.5))
        my_transforms.append(RandomBrightnessContrast(brightness_range=0.1, contrast_range=0.1, prob=0.5))
        # crop_size = int(img_size * random.uniform(0.8, 1.))
        # my_transforms.append(RandomCrop(crop_size=[crop_size, crop_size], out_size=(img_size, img_size), prob=0.5))
        my_transforms.append(LargeScaleJitter(output_size=img_size, aug_scale_min=0.8, aug_scale_max=1.2, prob=0.2))
        # my_transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2))
        my_transforms.append(RandomSharpenOrBlur(kernel_size=3, sigma_sharp=(0.8, 1.2), amount=(0.3, 0.8), sigma_blur=(0.8, 1.2), prob=0.2))

    my_transforms.append(Resize(size=[img_size, img_size]))

    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms), mask_cache=mask_cache)
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        dataloader = DataLoader(gos_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms),
                                        eval_ori_resolution=True, mask_cache=mask_cache)
            dataloader = DataLoader(gos_dataset, batch_size=batch_size, num_workers=0)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        mask_inputs = sample.get('mask_inputs', torch.zeros(1, image.shape[1], image.shape[2]))
        path = sample.get('path', None)

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
            edge = torch.flip(edge, dims=[2])
            mask_inputs = torch.flip(mask_inputs, dims=[2])

        result = {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': shape, 'mask_inputs': mask_inputs}
        if path is not None:
            result['path'] = path
        return result


class RandomBrightnessContrast(object):
    def __init__(self, brightness_range=0.2, contrast_range=0.2, prob=0.5):
        """
        随机调整图像的亮度和对比度
        
        Args:
            brightness_range: 亮度变化范围，实际变化值在 [-brightness_range, brightness_range] 之间
            contrast_range: 对比度变化范围，实际变化值在 [1-contrast_range, 1+contrast_range] 之间
            prob: 应用变换的概率
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        mask_inputs = sample.get('mask_inputs', torch.zeros(1, image.shape[1], image.shape[2]))
        path = sample.get('path', None)

        # 随机调整对比度
        if random.random() <= self.prob:
            contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            mean_value = image.mean()
            image = mean_value + (image - mean_value) * contrast_factor
            image = torch.clamp(image, 0, 255)
            
        # 随机调整亮度
        if random.random() <= self.prob:
            brightness_factor = random.uniform(-self.brightness_range, self.brightness_range) * image.max()
            image = image + brightness_factor
            image = torch.clamp(image, 0, 255)
        result = {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': shape, 'mask_inputs': mask_inputs}
        if path is not None:
            result['path'] = path
        return result


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        mask_inputs = sample.get('mask_inputs', torch.zeros(1, image.shape[1], image.shape[2]))
        path = sample.get('path', None)

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode='nearest'), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode='nearest'), dim=0)
        edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge, 0), self.size, mode='nearest'), dim=0)
        mask_inputs = torch.squeeze(F.interpolate(torch.unsqueeze(mask_inputs, 0), self.size, mode='nearest'), dim=0)

        result = {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': torch.tensor(self.size), 'mask_inputs': mask_inputs}
        if path is not None:
            result['path'] = path
        return result

class RandomSharpenOrBlur(object):
    def __init__(
        self,
        kernel_size=3,
        sigma_sharp=(0.5, 1.5),
        amount=(0.5, 1.0),
        sigma_blur=(0.5, 1.5),
        prob=0.2
    ):
        self.kernel_size = kernel_size
        self.sigma_sharp = sigma_sharp
        self.amount = amount
        self.sigma_blur = sigma_blur
        self.prob = prob

    def __call__(self, sample):
        imidx = sample['imidx']
        image = sample['image']
        label = sample['label']
        edge = sample['edge']
        shape = sample['shape']
        mask_inputs = sample.get(
            'mask_inputs',
            torch.zeros(1, image.shape[1], image.shape[2], device=image.device)
        )
        path = sample.get('path', None)

        if torch.rand(1) <= self.prob:
            # 0.5 sharpen, 0.5 blur
            if torch.rand(1) < 0.5:
                # --- Unsharp Mask ---
                sigma = torch.empty(1).uniform_(*self.sigma_sharp).item()
                amount = torch.empty(1).uniform_(*self.amount).item()
                blur = T.GaussianBlur(self.kernel_size, sigma)(image)
                image = torch.clamp(image + amount * (image - blur), 0.0, 1.0)
            else:
                # --- Gaussian Blur ---
                sigma = torch.empty(1).uniform_(*self.sigma_blur).item()
                image = T.GaussianBlur(self.kernel_size, sigma)(image)

        result = {
            'imidx': imidx,
            'image': image,
            'label': label,
            'edge': edge,
            'shape': shape,
            'mask_inputs': mask_inputs
        }
        if path is not None:
            result['path'] = path

        return result

class RandomCrop(object):
    def __init__(self, crop_size=[288, 288], out_size=(512, 512), prob=0.5):
        self.crop_size = crop_size
        self.out_size = out_size
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        if random.random()<self.prob:

            h, w = image.shape[1:]
            new_h, new_w = self.crop_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[:, top:top + new_h, left:left + new_w]
            label = label[:, top:top + new_h, left:left + new_w]
            edge = edge[:, top:top + new_h, left:left + new_w]

            image = F.interpolate(image.unsqueeze(0), self.out_size, mode='nearest').squeeze(0)
            label = F.interpolate(label.unsqueeze(0), self.out_size, mode='nearest').squeeze(0)
            edge = F.interpolate(edge.unsqueeze(0), self.out_size, mode='nearest').squeeze(0)

            return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': torch.tensor(self.out_size)}
        else:
            return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': shape}



class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': shape}


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=512, aug_scale_min=0.8, aug_scale_max=1.5, prob=0.5):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.prob = prob

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, edge, image_size = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        # resize keep ratio
        # out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()
        if random.random() < self.prob:
            random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
            scaled_size = (random_scale * self.desired_size).round()

            scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
            scaled_size = (image_size * scale).round().long()

            scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='nearest'),
                                         dim=0)
            scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='nearest'),
                                         dim=0)
            scaled_edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge, 0), scaled_size.tolist(), mode='nearest'),
                                         dim=0)

            # random crop
            crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

            margin_h = max(scaled_size[0] - crop_size[0], 0).item()
            margin_w = max(scaled_size[1] - crop_size[1], 0).item()
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

            scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
            scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]
            scaled_edge = scaled_edge[:, crop_y1:crop_y2, crop_x1:crop_x2]

            # pad
            padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
            padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
            image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=image.mean())
            label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)
            edge = F.pad(scaled_edge, [0, padding_w, 0, padding_h], value=0)

            return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': torch.tensor(image.shape[-2:])}
        else:
            return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': image_size}


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, mask_cache=None):

        self.transform = transform
        self.dataset = {}
        self.mask_cache = mask_cache  # MaskCache instance for loading previous epoch masks
        
        # combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per image
        im_name_list = []  # image name
        im_path_list = []  # im path
        gt_path_list = []  # gt path
        im_ext_list = []  # im ext
        gt_ext_list = []  # gt ext
        for i in range(0, len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend(
                [x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = Image.open(im_path).convert('RGB')
        im = np.array(im)
        gt = Image.open(gt_path)
        gt = np.array(gt)
        # im = io.imread(im_path)
        # gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        if im.shape[2] == 4:
            im = im[:, :, :3]

        if im.shape[:2] != gt.shape[:2]:
            gt.resize(im.shape[:2])


        # edge = cv2.Canny(gt, 100, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # 半径 5 -> 直径 11
        gt_dilated = cv2.dilate(gt, kernel, iterations=1)
        imgt = im*(gt>0)[:,:,None]
        edge = cv2.Canny(im, im.mean(), imgt[imgt>0].mean()-im.mean())
        blurred = cv2.GaussianBlur(edge, (3, 3), 0)
        edge = ((blurred-edge)>0).astype(np.float32) * (1 - gt / 255.)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        # tophat = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, kernel)
        # edge = (tophat[:,:,0] > min(imgt[imgt>0].mean()-im.mean(), 50)).astype(np.float32) * (1 - gt_dilated / 255.)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        edge = torch.unsqueeze(torch.tensor(edge, dtype=torch.float32), 0)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "edge": edge,
            "shape": torch.tensor(im.shape[-2:]),
            "path": self.dataset["im_path"][idx]
        }

        # 尝试从mask缓存中加载上一轮的预测结果
        if self.mask_cache is not None:
            cached_mask = self.mask_cache.get_mask(im_path)
            if cached_mask is not None:
                # 确保mask的尺寸与图像匹配
                if len(cached_mask.shape) == 2:
                    cached_mask = torch.unsqueeze(cached_mask, 0)  # 添加channel维度
                sample["mask_inputs"] = cached_mask
            else:
                # 如果没有缓存的mask，创建一个空的tensor
                sample["mask_inputs"] = torch.zeros(1, im.shape[1], im.shape[2])
        else:
            # 如果没有mask_cache，创建一个空的tensor
            sample["mask_inputs"] = torch.zeros(1, im.shape[1], im.shape[2])

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample
