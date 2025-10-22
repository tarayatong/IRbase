#!/usr/bin/env python3
"""
测试新的Neck组件架构
"""
import torch
import torch.nn as nn
import sys
import os

# 添加路径以便导入模块
sys.path.append('/home/tyf/code/IRSAM-main')

from segment_anything_training.modeling.IRSAM_encoder import Neck, ChannelShuffle

def test_channel_shuffle():
    """测试通道shuffle功能"""
    print("测试通道shuffle...")
    
    # 创建测试数据
    x = torch.randn(2, 8, 4, 4)  # [batch, channels, height, width]
    print(f"输入形状: {x.shape}")
    
    # 应用通道shuffle
    shuffle = ChannelShuffle(groups=2)
    y = shuffle(x)
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状不变
    assert y.shape == x.shape, f"形状不匹配: {y.shape} vs {x.shape}"
    print("✓ 通道shuffle测试通过")

def test_neck_architecture():
    """测试新的Neck架构"""
    print("\n测试新的Neck架构...")
    
    # 创建测试数据
    batch_size = 2
    in_channels = 64
    out_channels = 64
    height, width = 32, 32
    
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"输入形状: {x.shape}")
    
    # 创建Neck组件
    neck = Neck(in_dim=in_channels, out_dim=out_channels)
    
    # 前向传播
    y = neck(x)
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, out_channels, height//2, width//2)  # stride=2
    assert y.shape == expected_shape, f"输出形状不匹配: {y.shape} vs {expected_shape}"
    print("✓ Neck架构测试通过")

def test_neck_without_stride():
    """测试不带stride的Neck"""
    print("\n测试不带stride的Neck...")
    
    # 创建测试数据
    batch_size = 2
    in_channels = 128
    out_channels = 64
    height, width = 16, 16
    
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"输入形状: {x.shape}")
    
    # 创建Neck组件（stride=1）
    neck = Neck(in_dim=in_channels, out_dim=out_channels, stride=1)
    
    # 前向传播
    y = neck(x)
    print(f"输出形状: {y.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, out_channels, height, width)
    assert y.shape == expected_shape, f"输出形状不匹配: {y.shape} vs {expected_shape}"
    print("✓ 不带stride的Neck测试通过")

def test_gradient_flow():
    """测试梯度流"""
    print("\n测试梯度流...")
    
    # 创建测试数据
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    neck = Neck(in_dim=64, out_dim=64, stride=1)
    
    # 前向传播
    y = neck(x)
    
    # 计算损失
    loss = y.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    assert x.grad is not None, "输入梯度为None"
    assert x.grad.shape == x.shape, f"梯度形状不匹配: {x.grad.shape} vs {x.shape}"
    
    # 检查梯度是否包含有效值
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0, f"梯度范数为0: {grad_norm}"
    
    print(f"✓ 梯度流测试通过，梯度范数: {grad_norm:.6f}")

if __name__ == "__main__":
    print("开始测试新的Neck组件...")
    
    try:
        test_channel_shuffle()
        test_neck_architecture()
        test_neck_without_stride()
        test_gradient_flow()
        
        print("\n🎉 所有测试通过！新的Neck组件工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
