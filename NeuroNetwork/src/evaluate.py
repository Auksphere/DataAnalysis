"""
评估模块 - 包含模型评估和分析功能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
from config import Config

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, config=None):
        self.config = config or Config()
        self.model = model.to(self.config.DEVICE)
        self.model.eval()
    
    def evaluate(self, testloader):
        """评估模型性能"""
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc='Evaluating'):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        return accuracy, all_predictions, all_targets
    
    def plot_confusion_matrix(self, predictions, targets, model_name='Model'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.CLASS_NAMES,
                   yticklabels=self.config.CLASS_NAMES)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(f'{self.config.RESULTS_PATH}/plots', exist_ok=True)
        plt.savefig(f'{self.config.RESULTS_PATH}/plots/{model_name}_confusion_matrix.png')
        plt.show()
    
    def get_classification_report(self, predictions, targets):
        """获取分类报告"""
        return classification_report(
            targets, predictions,
            target_names=self.config.CLASS_NAMES,
            output_dict=True
        )
    
    def analyze_class_performance(self, predictions, targets, model_name='Model'):
        """分析各类别性能"""
        report = self.get_classification_report(predictions, targets)
        
        # 提取各类别的指标
        classes = self.config.CLASS_NAMES
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]
        
        # 绘制柱状图
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Per-Class Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_PATH}/plots/{model_name}_class_performance.png')
        plt.show()
        
        return report
    
    def visualize_feature_maps(self, testloader, layer_name='conv1', num_samples=1):
        """可视化特征图"""
        # 注册钩子函数来捕获中间层输出
        activations = {}
        def hook_fn(module, input, output):
            activations[layer_name] = output
        
        # 查找指定层并注册钩子
        for name, module in self.model.named_modules():
            if layer_name in name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        # 获取一个批次的数据
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images = images[:num_samples].to(self.config.DEVICE)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(images)
        
        # 获取特征图
        if layer_name in activations:
            feature_maps = activations[layer_name].cpu().numpy()
            
            # 可视化第一个样本的前16个特征图
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(min(16, feature_maps.shape[1])):
                axes[i].imshow(feature_maps[0, i], cmap='viridis')
                axes[i].set_title(f'Feature Map {i}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.config.RESULTS_PATH}/plots/feature_maps_{layer_name}.png')
            plt.show()
        
        # 清除钩子
        handle.remove()
    
    def compare_models(self, results_dict):
        """比较多个模型的性能"""
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['accuracy'] for model in models]
        
        # 绘制准确率比较图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        
        # 在柱子上添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('CNN Model Accuracy Comparison')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_PATH}/plots/model_comparison.png')
        plt.show()
    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs, model_name):
        """绘制训练曲线"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_title(f'{model_name} - Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
        ax2.set_title(f'{model_name} - Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.RESULTS_PATH}/plots/{model_name}_training_curves.png')
        plt.show()
    
    def predict_single_image(self, image_tensor, class_names):
        """预测单张图片"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.config.DEVICE)
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]