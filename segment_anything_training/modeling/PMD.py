import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
from timm.layers import DropPath

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 如果输入通道和输出通道不一致，添加一个卷积层进行转换
        if in_channels != out_channels:
            self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        else:
            self.channel_conv = None
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, out_channels, out_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=out_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, out_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(out_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=out_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        # 如果需要调整通道数
        if self.channel_conv is not None:
            x = self.channel_conv(x)
            
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        print(f"Get_gradient_nopadding输入: {x.shape}, 设备: {x.device}")
        # 优化：避免逐通道处理，使用分组卷积
        B, C, H, W = x.shape
        
        # 将所有通道展开为batch维度进行并行处理
        x_reshaped = x.view(B * C, 1, H, W)
        
        # 使用分组卷积并行处理所有通道
        weight_v_expanded = self.weight_v.repeat(C, 1, 1, 1)
        weight_h_expanded = self.weight_h.repeat(C, 1, 1, 1)
        
        x_v = F.conv2d(x_reshaped, weight_v_expanded, padding=1, groups=C)
        x_h = F.conv2d(x_reshaped, weight_h_expanded, padding=1, groups=C)
        
        # 计算梯度幅度
        x_grad = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        
        # 重新reshape回原来的形状
        x_grad = x_grad.view(B, C, H, W)
        print(f"Get_gradient_nopadding输出: {x_grad.shape}, 设备: {x_grad.device}")
        
        return x_grad


class Get_curvature(nn.Module):
    def __init__(self):
        super(Get_curvature, self).__init__()
        kernel_v1 = [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]]
        kernel_h1 = [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]]
        kernel_h2 = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, -2, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        kernel_v2 = [[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -2, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]]
        kernel_w2 = [[1, 0, -1],
                     [0, 0, 0],
                     [-1, 0, 1]]
        kernel_h1 = torch.FloatTensor(kernel_h1).unsqueeze(0).unsqueeze(0)
        kernel_v1 = torch.FloatTensor(kernel_v1).unsqueeze(0).unsqueeze(0)
        kernel_v2 = torch.FloatTensor(kernel_v2).unsqueeze(0).unsqueeze(0)
        kernel_h2 = torch.FloatTensor(kernel_h2).unsqueeze(0).unsqueeze(0)
        kernel_w2 = torch.FloatTensor(kernel_w2).unsqueeze(0).unsqueeze(0)
        self.weight_h1 = nn.Parameter(data=kernel_h1, requires_grad=False)
        self.weight_v1 = nn.Parameter(data=kernel_v1, requires_grad=False)
        self.weight_v2 = nn.Parameter(data=kernel_v2, requires_grad=False)
        self.weight_h2 = nn.Parameter(data=kernel_h2, requires_grad=False)
        self.weight_w2 = nn.Parameter(data=kernel_w2, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v1, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h1, padding=1)
            x_i_v2 = F.conv2d(x_i.unsqueeze(1), self.weight_v2, padding=2)
            x_i_h2 = F.conv2d(x_i.unsqueeze(1), self.weight_h2, padding=2)
            x_i_w2 = F.conv2d(x_i.unsqueeze(1), self.weight_w2, padding=1)
            sum = torch.pow((torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2)), 3 / 2)
            fg = torch.mul(torch.pow(x_i_v, 2), x_i_v2) + 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_h2)
            fh = torch.mul(torch.pow(x_i_v, 2), x_i_h2) - 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_v2)
            x_i = torch.div(torch.abs(fg - fh), sum + 1e-10)
            x_i = torch.div(torch.abs(fh), sum + 1e-10)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, out_dims):
        super(FeatureEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, out_dims[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dims[0], out_dims[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(out_dims[0], out_dims[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dims[1], out_dims[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(out_dims[1], out_dims[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(out_dims[2], out_dims[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x1 = x

        # Stage 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x2 = x

        # Stage 3
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x3 = x

        # Stage 4
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x4 = x

        return x1, x2, x3, x4


class PMD_features(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(PMD_features, self).__init__()
        # self.PMD_head = Get_curvature()
        self.wavelet_decomp = MBWTConv2d(in_dims, out_dims, stride=4)
        self.PMD_head = Get_gradient_nopadding()
        # 添加1x1卷积控制通道数为相加的一半
        self.channel_reduce = nn.Conv2d(out_dims, out_dims // 2, kernel_size=1)
        # self.feature_ext = FeatureEncoder(out_dims)
        
        # 为所有参数设置lr_scale属性
        self._set_lr_scale()
        
    def _set_lr_scale(self, scale=1.0):
        """为所有参数设置lr_scale属性"""
        for name, p in self.named_parameters():
            p.lr_scale = scale
            p.param_name = name

    def forward(self, images):
        print(f"PMD输入形状: {images.shape}, 设备: {images.device}")
        wavelet_images = self.wavelet_decomp(images)
        print(f"小波分解后形状: {wavelet_images.shape}, 设备: {wavelet_images.device}")
        PMD_images = self.PMD_head(wavelet_images)
        print(f"PMD处理后形状: {PMD_images.shape}, 设备: {PMD_images.device}")
        # 通道数减半
        PMD_images = self.channel_reduce(PMD_images)
        print(f"通道减半后形状: {PMD_images.shape}, 设备: {PMD_images.device}")
        # PMD_feature = self.feature_ext(PMD_images)

        return PMD_images

# class Adapter(nn.Module):
#     def __init__(self, out_dims):
#         super(Adapter, self).__init__()
#         self.PMD_head = Get_gradient_nopadding()
#         self.feature_ext = FeatureEncoder(out_dims)
#
#     def forward(self, images):
#         PMD_images = self.PMD_head(images)
#         PMD_feature = self.feature_ext(PMD_images)
#
#         return PMD_feature
