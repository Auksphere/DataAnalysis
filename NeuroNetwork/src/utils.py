"""
工具函数模块
"""

import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import os
from config import Config

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config):
    """创建必要的目录"""
    directories = [
        config.MODEL_SAVE_PATH,
        config.RESULTS_PATH,
        f'{config.RESULTS_PATH}/plots',
        f'{config.RESULTS_PATH}/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def configure_chinese_font(prefer: str = 'auto'):
    """
    配置Matplotlib中文字体
    """
    candidates = [
        # Google Noto 字体
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        # 文泉驿
        'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        # 常见中文字体
        'SimHei', 'SimSun', 'Microsoft YaHei', 'STHeiti',
        # 退路：DejaVu Sans 支持部分符号
        'DejaVu Sans'
    ]

    if prefer != 'auto':
        candidates = [prefer] + [f for f in candidates if f != prefer]

    available = set(f.name for f in font_manager.fontManager.ttflist)
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    # 设置字体与负号
    plt.rcParams['axes.unicode_minus'] = False
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen]
        # 也设置到全局字体族，增强兼容
        plt.rcParams['font.family'] = 'sans-serif'
        try:
            print(f"已配置中文字体: {chosen}")
        except Exception:
            pass
    else:
        try:
            print("警告: 未检测到常见中文字体")
        except Exception:
            pass

def save_model_summary(model, model_name, config):
    """保存模型摘要信息"""
    summary_path = f'{config.RESULTS_PATH}/reports/{model_name}_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: {model_name}\n")
        f.write(f"模型结构:\n")
        f.write(str(model))
        f.write(f"\n\n参数数量: {count_parameters(model):,}\n")
        
        # 计算模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size = (param_size + buffer_size) / 1024 / 1024  # MB
        
        f.write(f"模型大小: {model_size:.2f} MB\n")

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def denormalize_image(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """反标准化图像"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def visualize_predictions(model, testloader, config, num_samples=8):
    """可视化预测结果"""
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        images_gpu = images.to(config.DEVICE)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # 反标准化图像
        img = denormalize_image(images[i])
        img = img.permute(1, 2, 0)
        
        # 显示图像
        axes[i].imshow(img)
        
        # 设置标题
        true_label = config.CLASS_NAMES[labels[i]]
        pred_label = config.CLASS_NAMES[predicted[i]]
        color = 'green' if labels[i] == predicted[i] else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/plots/prediction_examples.png')
    plt.show()

def plot_kernel_comparison(results_dict, config):
    """绘制不同卷积核大小的比较结果"""
    kernel_types = list(results_dict.keys())
    accuracies = [results_dict[k]['accuracy'] for k in kernel_types]
    
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = plt.bar(kernel_types, accuracies, color=colors[:len(kernel_types)])
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Kernel Size Impact on Performance')
    plt.ylim(0, max(accuracies) + 5)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加参考线
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                alpha=0.7, label=f'Avg Accuracy: {np.mean(accuracies):.2f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/plots/kernel_comparison.png')
    plt.show()

def generate_performance_report(results_dict, config):
    """生成性能报告"""
    report_path = f'{config.RESULTS_PATH}/reports/performance_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CNN图像分类性能报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 模型比较
        f.write("模型性能对比:\n")
        f.write("-" * 30 + "\n")
        for model_name, results in results_dict.items():
            f.write(f"{model_name}:\n")
            f.write(f"  准确率: {results['accuracy']:.2f}%\n")
            if 'parameters' in results:
                f.write(f"  参数数量: {results['parameters']:,}\n")
            f.write("\n")
        
        # 最佳模型
        best_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"最佳模型: {best_model[0]}\n")
        f.write(f"最高准确率: {best_model[1]['accuracy']:.2f}%\n\n")
        
        # 分析结论
        f.write("分析结论:\n")
        f.write("-" * 30 + "\n")
        accuracies = [results['accuracy'] for results in results_dict.values()]
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        f.write(f"平均准确率: {avg_acc:.2f}%\n")
        f.write(f"标准差: {std_acc:.2f}%\n")
        f.write(f"最大差异: {max(accuracies) - min(accuracies):.2f}%\n")

def print_training_info(config):
    """打印训练信息"""
    print("=" * 60)
    print("CNN图像分类项目")
    print("=" * 60)
    print(f"数据集: CIFAR-10")
    print(f"图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"类别数: {config.NUM_CLASSES}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    print(f"设备: {config.DEVICE}")
    print("-" * 60)