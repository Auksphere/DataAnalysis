"""
CNN模型定义 - 包含多种不同的卷积神经网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class BasicCNN(nn.Module):
    """基础CNN模型 - 使用3x3卷积核"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(BasicCNN, self).__init__()
        
        # 第一个卷积块 (32x32 -> 16x16)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二个卷积块 (16x16 -> 8x8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三个卷积块 (8x8 -> 4x4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 卷积层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 256 * 4 * 4)
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class LargeKernelCNN(nn.Module):
    """大卷积核CNN模型 - 使用7x7, 5x5卷积核"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(LargeKernelCNN, self).__init__()
        
        # 第一个卷积块 - 大卷积核
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二个卷积块 - 中等卷积核
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三个卷积块 - 小卷积核
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 256 * 4 * 4)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCNN(nn.Module):
    """ResNet风格的CNN模型"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNetCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DepthwiseSeparableCNN(nn.Module):
    """深度可分离卷积CNN模型"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(DepthwiseSeparableCNN, self).__init__()
        
        # 深度可分离卷积块
        self.depthwise1 = nn.Conv2d(input_channels, input_channels, 
                                   kernel_size=3, padding=1, groups=input_channels)
        self.pointwise1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.depthwise2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pointwise2 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.depthwise3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.pointwise3 = nn.Conv2d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 分类器
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 第一个深度可分离卷积块
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.pool1(F.relu(self.bn1(x)))
        
        # 第二个深度可分离卷积块
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.pool2(F.relu(self.bn2(x)))
        
        # 第三个深度可分离卷积块
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.pool3(F.relu(self.bn3(x)))
        
        # 分类器
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def get_model(model_type='basic', num_classes=10, input_channels=3):
    """根据类型返回相应的模型"""
    models = {
        'basic': BasicCNN,
        'large_kernel': LargeKernelCNN,
        'resnet': ResNetCNN,
        'depthwise': DepthwiseSeparableCNN
    }
    
    if model_type in models:
        return models[model_type](num_classes, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)