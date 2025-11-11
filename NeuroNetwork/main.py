"""
主程序 - CNN图像分类项目入口
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.data_loader import CIFAR10DataLoader
from src.cnn_models import get_model, count_parameters
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import *

def main():
    """主函数"""
    # 设置配置
    config = Config()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建目录
    create_directories(config)
    
    # 打印训练信息
    print_training_info(config)
    
    # 加载数据
    print("正在加载CIFAR-10数据集...")
    data_loader = CIFAR10DataLoader(config)
    trainloader, testloader = data_loader.load_data()
    
    # 显示样本图像
    print("显示样本图像...")
    data_loader.show_sample_images(testloader)
    
    # 获取数据统计信息
    print("\n数据集统计信息:")
    data_loader.get_data_statistics(trainloader)
    
    # 定义要测试的模型
    models_to_test = {
        'BasicCNN_3x3': 'basic',
        'LargeKernelCNN_7x5x3': 'large_kernel',
        'ResNetCNN': 'resnet',
        'DepthwiseCNN': 'depthwise'
    }
    
    # 存储结果
    results = {}
    
    # 训练和评估每个模型
    for model_name, model_type in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        # 创建模型
        model = get_model(model_type, config.NUM_CLASSES, config.INPUT_CHANNELS)
        
        # 保存模型摘要
        save_model_summary(model, model_name, config)
        
        # 创建训练器
        trainer = ModelTrainer(model, config)
        
        # 训练模型
        train_losses, train_accs, val_losses, val_accs = trainer.train(
            trainloader, testloader, model_name
        )
        
        # 评估模型
        evaluator = ModelEvaluator(model, config)
        accuracy, predictions, targets = evaluator.evaluate(testloader)
        
        # 绘制训练曲线
        evaluator.plot_training_curves(train_losses, train_accs, val_losses, val_accs, model_name)
        
        # 绘制混淆矩阵
        evaluator.plot_confusion_matrix(predictions, targets, model_name)
        
        # 分析类别性能
        class_report = evaluator.analyze_class_performance(predictions, targets, model_name)
        
        # 可视化特征图
        try:
            evaluator.visualize_feature_maps(testloader, 'conv1', 1)
        except:
            print("无法可视化特征图")
        
        # 存储结果
        results[model_name] = {
            'accuracy': accuracy,
            'parameters': count_parameters(model),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'class_report': class_report
        }
        
        print(f"模型 {model_name} 训练完成，测试准确率: {accuracy:.2f}%")
    
    # 比较所有模型
    print(f"\n{'='*60}")
    print("模型性能比较")
    print(f"{'='*60}")
    
    evaluator = ModelEvaluator(model, config)  # 使用最后一个模型的评估器
    evaluator.compare_models(results)
    
    # 绘制卷积核比较图
    plot_kernel_comparison(results, config)
    
    # 生成性能报告
    generate_performance_report(results, config)
    
    # 可视化预测结果
    visualize_predictions(model, testloader, config)
    
    # 打印最终结果
    print("\n最终结果:")
    print("-" * 40)
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.2f}% (参数: {result['parameters']:,})")
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.2f}%)")
    
    print(f"\n所有结果已保存到 {config.RESULTS_PATH} 目录")
    print("训练完成!")

if __name__ == '__main__':
    main()