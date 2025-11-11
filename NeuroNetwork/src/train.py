"""
训练模块 - 包含模型训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
from config import Config

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, config=None):
        self.config = config or Config()
        self.model = model.to(self.config.DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=15, gamma=0.1
        )
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, trainloader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(trainloader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, testloader):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(testloader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(val_bar):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(testloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, trainloader, testloader, model_name='model'):
        """完整训练过程"""
        print(f"开始训练模型: {model_name}")
        print(f"设备: {self.config.DEVICE}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f'\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(trainloader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(testloader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(model_name, epoch, val_acc)
                print(f'保存最佳模型，验证准确率: {best_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'\n训练完成! 总时间: {total_time/60:.2f} 分钟')
        print(f'最佳验证准确率: {best_acc:.2f}%')
        
        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies
    
    def save_model(self, model_name, epoch, accuracy):
        """保存模型"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{model_name}_best.pth')
        torch.save(checkpoint, save_path)
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"模型已加载，准确率: {checkpoint['accuracy']:.2f}%")
        
        return checkpoint['accuracy']