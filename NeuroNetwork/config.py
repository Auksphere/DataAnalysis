"""
配置文件 - 包含所有超参数和设置
"""

import torch

class Config:
    # 数据相关
    DATA_PATH = './data'  # 在NeuroNetwork目录下
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # 训练相关
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 模型相关
    NUM_CLASSES = 10
    INPUT_CHANNELS = 3
    IMAGE_SIZE = 32
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    MODEL_SAVE_PATH = './models'
    RESULTS_PATH = './results'
    
    # CIFAR-10类别名称
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # 不同卷积核配置
    KERNEL_CONFIGS = {
        'small': [3, 3, 3],      # 小卷积核
        'medium': [5, 5, 5],     # 中等卷积核
        'large': [7, 5, 3],      # 大卷积核
        'mixed': [7, 5, 3]       # 混合卷积核
    }