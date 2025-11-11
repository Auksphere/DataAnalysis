"""
数据加载模块 - 处理CIFAR-10数据集的加载和预处理
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from config import Config

class CIFAR10DataLoader:
    """CIFAR-10数据加载器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        # 数据增强和标准化
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    def load_data(self):
        """加载训练和测试数据"""
        # 下载并加载训练数据
        trainset = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, 
            train=True,
            download=True, 
            transform=self.train_transform
        )
        
        trainloader = DataLoader(
            trainset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True, 
            num_workers=self.config.NUM_WORKERS
        )
        
        # 下载并加载测试数据
        testset = torchvision.datasets.CIFAR10(
            root=self.config.DATA_PATH, 
            train=False,
            download=True, 
            transform=self.test_transform
        )
        
        testloader = DataLoader(
            testset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS
        )
        
        return trainloader, testloader
    
    def show_sample_images(self, dataloader, num_samples=8):
        """显示样本图像"""
        # 获取一个批次的数据
        dataiter = iter(dataloader)
        images, labels = next(dataiter)
        
        # 反标准化
        def denormalize(tensor):
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor
        
        # 创建图形
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = denormalize(images[i].clone())
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0)
            
            axes[i].imshow(img)
            axes[i].set_title(f'{self.config.CLASS_NAMES[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_PATH}/plots/sample_images.png')
        plt.show()
    
    def get_data_statistics(self, trainloader):
        """获取数据统计信息"""
        total_samples = len(trainloader.dataset)
        class_counts = torch.zeros(self.config.NUM_CLASSES)
        
        for _, labels in trainloader:
            for label in labels:
                class_counts[label] += 1
        
        print(f"总样本数: {total_samples}")
        print(f"训练批次数: {len(trainloader)}")
        print(f"每个类别的样本数:")
        for i, count in enumerate(class_counts):
            print(f"  {self.config.CLASS_NAMES[i]}: {int(count)}")
        
        return class_counts