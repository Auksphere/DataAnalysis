"""
LDA和决策树分类方法对比分析 - 主程序
使用Wine数据集进行分类分析和方法对比
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import WineDataLoader
from src.lda_classifier import LDAClassifier
from src.decision_tree_classifier import DecisionTreeClassifierCustom
from src.comparison_analysis import ComparisonAnalysis

def main():
    """主函数"""
    print("=" * 60)
    print("LDA vs 决策树分类方法对比分析")
    print("数据集: Wine Quality Dataset")
    print("=" * 60)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 1. 数据加载和预处理
    print("\n【步骤1】数据加载和预处理")
    print("-" * 40)
    
    loader = WineDataLoader(test_size=0.3, random_state=42)
    
    # 加载数据
    df = loader.load_data()
    
    # 探索性数据分析
    loader.explore_data(save_path='results')
    
    # 预处理数据
    data = loader.preprocess_data()
    
    # 保存数据
    loader.save_processed_data(save_path='data')
    
    # 获取数据信息
    data_info = loader.get_data_info()
    
    # 2. LDA分类分析
    print("\n【步骤2】LDA分类分析")
    print("-" * 40)
    
    # 创建和训练LDA分类器
    lda_classifier = LDAClassifier(n_components=2, solver='svd')
    lda_classifier.train(
        data['X_train'], 
        data['y_train'],
        feature_names=data_info['feature_names'],
        target_names=data_info['target_names']
    )
    
    # LDA预测和评估
    lda_results = lda_classifier.predict(data['X_test'], data['y_test'])
    
    # LDA可视化
    lda_classifier.visualize_results(data['X_test'], data['y_test'], lda_results, 'results')
    
    # LDA决策边界分析
    lda_classifier.analyze_decision_boundary(data['X_test'], data['y_test'], 'results')
    
    # 3. 决策树分类分析
    print("\n【步骤3】决策树分类分析")
    print("-" * 40)
    
    # 创建和训练决策树分类器
    dt_classifier = DecisionTreeClassifierCustom(
        criterion='gini', 
        max_depth=10, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_classifier.train(
        data['X_train'], 
        data['y_train'],
        feature_names=data_info['feature_names'],
        target_names=data_info['target_names']
    )
    
    # 决策树预测和评估
    dt_results = dt_classifier.predict(data['X_test'], data['y_test'])
    
    # 决策树可视化
    dt_classifier.visualize_results(data['X_test'], data['y_test'], dt_results, 'results')
    
    # 过拟合分析
    dt_classifier.analyze_overfitting(
        data['X_train'], data['y_train'], 
        data['X_test'], data['y_test'], 
        'results'
    )
    
    # 4. 对比分析
    print("\n【步骤4】LDA vs 决策树对比分析")
    print("-" * 40)
    
    # 创建对比分析器
    comparison = ComparisonAnalysis(
        lda_classifier.model,
        dt_classifier.model,
        feature_names=data_info['feature_names'],
        target_names=data_info['target_names']
    )
    
    # 性能对比
    performance_comparison = comparison.compare_performance(
        data['X_test'], data['y_test'],
        data['X_train'], data['y_train']
    )
    
    # 特征重要性对比
    feature_importance_comparison = comparison.analyze_feature_importance()
    
    # 可视化对比结果
    comparison.visualize_comparison(data['X_test'], data['y_test'], 'results')
    
    # 生成分析总结
    summary = comparison.generate_analysis_summary()
    
    # 5. 生成综合报告
    print("\n【步骤5】生成综合分析报告")
    print("-" * 40)
    
    generate_comprehensive_report(
        data_info, 
        lda_classifier, 
        dt_classifier, 
        performance_comparison,
        summary
    )
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("结果文件保存在 'results/' 目录中")
    print("数据文件保存在 'data/' 目录中")
    print("详细报告: Classification_Analysis_Report.md")
    print("=" * 60)

def generate_comprehensive_report(data_info, lda_classifier, dt_classifier, performance_comparison, summary):
    """生成综合分析报告"""
    
    report_path = 'Classification_Analysis_Report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LDA vs 决策树分类方法对比分析报告\n\n")
        
        # 1. 项目概述
        f.write("## 1. 项目概述\n\n")
        f.write("本项目实现并比较了两种经典的机器学习分类方法：\n")
        f.write("- **线性判别分析 (Linear Discriminant Analysis, LDA)**\n")
        f.write("- **决策树 (Decision Tree)**\n\n")
        f.write("使用Wine质量数据集进行实验，该数据集包含178个样本和13个特征，分为3个类别。\n\n")
        
        # 2. 数据集信息
        f.write("## 2. 数据集信息\n\n")
        f.write(f"- **数据集名称**: Wine Quality Dataset\n")
        f.write(f"- **样本数量**: {data_info['n_samples']}\n")
        f.write(f"- **特征数量**: {data_info['n_features']}\n")
        f.write(f"- **类别数量**: {data_info['n_classes']}\n")
        f.write(f"- **特征类型**: 连续数值型特征\n")
        f.write(f"- **目标类别**: {', '.join(data_info['target_names'])}\n\n")
        
        f.write("### 主要特征包括:\n")
        for i, feature in enumerate(data_info['feature_names'][:10], 1):
            f.write(f"{i}. {feature}\n")
        if len(data_info['feature_names']) > 10:
            f.write(f"... 等共{len(data_info['feature_names'])}个特征\n")
        f.write("\n")
        
        # 3. 方法介绍
        f.write("## 3. 方法介绍\n\n")
        f.write("### 3.1 线性判别分析 (LDA)\n\n")
        f.write("LDA是一种经典的线性分类方法，主要特点：\n")
        f.write("- **原理**: 寻找最佳的线性组合来分离不同类别\n")
        f.write("- **假设**: 各类别数据服从多元正态分布，协方差矩阵相同\n")
        f.write("- **优势**: 计算简单，适合线性可分数据，具有降维功能\n")
        f.write("- **局限**: 对非线性关系建模能力有限，对异常值敏感\n\n")
        
        f.write("### 3.2 决策树 (Decision Tree)\n\n")
        f.write("决策树是一种基于树结构的分类方法，主要特点：\n")
        f.write("- **原理**: 通过递归分割特征空间构建决策规则\n")
        f.write("- **分割标准**: 使用基尼不纯度或信息熵选择最佳分割\n")
        f.write("- **优势**: 可解释性强，能处理非线性关系，对数据预处理要求低\n")
        f.write("- **局限**: 容易过拟合，对噪声敏感，预测结果不稳定\n\n")
        
        # 4. 实验结果
        f.write("## 4. 实验结果\n\n")
        f.write("### 4.1 模型配置\n\n")
        
        # LDA配置
        lda_info = lda_classifier.get_model_info()
        f.write("**LDA配置:**\n")
        f.write(f"- 降维组件数: {lda_info.get('n_components', 'N/A')}\n")
        f.write(f"- 求解器: {lda_info.get('solver', 'N/A')}\n")
        if 'explained_variance_ratio' in lda_info:
            f.write(f"- 解释方差比: {lda_info['explained_variance_ratio']}\n")
        f.write("\n")
        
        # 决策树配置
        dt_info = dt_classifier.get_model_info()
        f.write("**决策树配置:**\n")
        f.write(f"- 分割标准: {dt_info.get('criterion', 'N/A')}\n")
        f.write(f"- 最大深度限制: {dt_info.get('max_depth', 'N/A')}\n")
        f.write(f"- 实际树深度: {dt_info.get('actual_depth', 'N/A')}\n")
        f.write(f"- 叶节点数: {dt_info.get('n_leaves', 'N/A')}\n")
        f.write(f"- 节点总数: {dt_info.get('n_nodes', 'N/A')}\n")
        f.write("\n")
        
        # 4.2 性能对比
        f.write("### 4.2 性能对比\n\n")
        f.write("| 指标 | LDA | 决策树 | 差异 |\n")
        f.write("|------|-----|--------|------|\n")
        
        lda_perf = performance_comparison['LDA']
        dt_perf = performance_comparison['Decision_Tree']
        
        f.write(f"| 准确率 | {lda_perf['accuracy']:.4f} | {dt_perf['accuracy']:.4f} | {abs(lda_perf['accuracy'] - dt_perf['accuracy']):.4f} |\n")
        f.write(f"| 精确率 | {lda_perf['precision']:.4f} | {dt_perf['precision']:.4f} | {abs(lda_perf['precision'] - dt_perf['precision']):.4f} |\n")
        f.write(f"| 召回率 | {lda_perf['recall']:.4f} | {dt_perf['recall']:.4f} | {abs(lda_perf['recall'] - dt_perf['recall']):.4f} |\n")
        f.write(f"| F1分数 | {lda_perf['f1_score']:.4f} | {dt_perf['f1_score']:.4f} | {abs(lda_perf['f1_score'] - dt_perf['f1_score']):.4f} |\n")
        f.write(f"| AUC | {lda_perf['auc']:.4f} | {dt_perf['auc']:.4f} | {abs(lda_perf['auc'] - dt_perf['auc']):.4f} |\n")
        f.write(f"| 预测时间(s) | {lda_perf['prediction_time']:.6f} | {dt_perf['prediction_time']:.6f} | {abs(lda_perf['prediction_time'] - dt_perf['prediction_time']):.6f} |\n\n")
        
        # 5. 优势和局限性分析
        f.write("## 5. 方法优势和局限性分析\n\n")
        
        f.write("### 5.1 LDA (线性判别分析)\n\n")
        f.write("**优势:**\n")
        f.write("1. **计算效率高**: 线性模型，训练和预测速度快\n")
        f.write("2. **降维功能**: 同时进行分类和降维，减少特征空间维度\n")
        f.write("3. **理论基础扎实**: 基于贝叶斯决策理论，有明确的统计意义\n")
        f.write("4. **参数少**: 模型复杂度低，不容易过拟合\n")
        f.write("5. **内存需求小**: 模型参数少，存储空间需求小\n\n")
        
        f.write("**局限性:**\n")
        f.write("1. **线性假设**: 只能处理线性可分的问题，对非线性关系建模能力有限\n")
        f.write("2. **分布假设**: 假设数据服从多元正态分布，现实中往往不满足\n")
        f.write("3. **协方差假设**: 假设各类别协方差矩阵相同，限制了应用范围\n")
        f.write("4. **异常值敏感**: 对离群点和噪声比较敏感\n")
        f.write("5. **特征选择**: 需要人工进行特征工程和选择\n\n")
        
        f.write("### 5.2 决策树 (Decision Tree)\n\n")
        f.write("**优势:**\n")
        f.write("1. **可解释性强**: 决策过程直观，易于理解和解释\n")
        f.write("2. **非线性建模**: 能够捕获特征间的非线性关系和交互作用\n")
        f.write("3. **数据预处理简单**: 不需要标准化，能处理数值和类别特征\n")
        f.write("4. **特征选择自动**: 自动选择重要特征进行分割\n")
        f.write("5. **处理缺失值**: 能够处理含有缺失值的数据\n\n")
        
        f.write("**局限性:**\n")
        f.write("1. **容易过拟合**: 特别是树很深时，泛化能力差\n")
        f.write("2. **不稳定性**: 数据的小变化可能导致完全不同的树结构\n")
        f.write("3. **偏向性**: 倾向于选择取值较多的特征进行分割\n")
        f.write("4. **线性关系建模弱**: 对于简单的线性关系，可能构建复杂的树\n")
        f.write("5. **预测连续值困难**: 对回归问题的处理不如分类问题\n\n")
        
        # 6. 实验总结和建议
        f.write("## 6. 实验总结和建议\n\n")
        
        if summary:
            f.write("### 6.1 性能总结\n\n")
            f.write(f"- **最佳模型**: {summary['best_accuracy_model']}\n")
            f.write(f"- **最佳准确率**: {summary['best_accuracy']:.4f}\n")
            f.write(f"- **准确率差异**: {summary['accuracy_difference']:.4f}\n")
            f.write(f"- **速度更快的模型**: {summary['faster_model']}\n")
            f.write(f"- **时间差异**: {summary['time_difference']:.6f} 秒\n\n")
        
        f.write("### 6.2 应用建议\n\n")
        f.write("**选择LDA的情况:**\n")
        f.write("- 数据维度较高，需要降维\n")
        f.write("- 特征间关系主要是线性的\n")
        f.write("- 对计算效率有较高要求\n")
        f.write("- 数据近似满足正态分布假设\n")
        f.write("- 各类别样本量相对均衡\n\n")
        
        f.write("**选择决策树的情况:**\n")
        f.write("- 需要模型的可解释性\n")
        f.write("- 特征间存在复杂的非线性关系\n")
        f.write("- 数据包含类别特征和数值特征\n")
        f.write("- 数据预处理资源有限\n")
        f.write("- 需要自动进行特征选择\n\n")
        
        f.write("### 6.3 改进方向\n\n")
        f.write("**LDA改进:**\n")
        f.write("- 使用核LDA处理非线性问题\n")
        f.write("- 结合特征选择方法提高性能\n")
        f.write("- 使用正则化LDA处理小样本问题\n\n")
        
        f.write("**决策树改进:**\n")
        f.write("- 使用集成方法(如随机森林、XGBoost)提高稳定性\n")
        f.write("- 调整超参数防止过拟合\n")
        f.write("- 结合剪枝技术优化树结构\n\n")
        
        # 7. 结论
        f.write("## 7. 结论\n\n")
        f.write("通过对Wine数据集的分析，我们得出以下结论：\n\n")
        f.write("1. **性能表现**: 两种方法在该数据集上都取得了不错的分类效果\n")
        f.write("2. **适用场景**: LDA适合线性可分问题，决策树适合复杂非线性问题\n")
        f.write("3. **计算效率**: LDA在预测速度上通常更快\n")
        f.write("4. **可解释性**: 决策树提供了更直观的决策过程\n")
        f.write("5. **鲁棒性**: 各有优缺点，需要根据具体应用场景选择\n\n")
        
        f.write("本实验展示了机器学习中\"没有免费的午餐\"定理 - 没有一种算法在所有问题上都是最优的。")
        f.write("实际应用中应该根据数据特性、业务需求和计算资源来选择合适的方法。\n\n")
        
        # 8. 附录
        f.write("## 8. 附录\n\n")
        f.write("### 相关文件\n")
        f.write("- `data/`: 数据文件目录\n")
        f.write("- `results/`: 结果和图表目录\n")
        f.write("- `src/`: 源代码目录\n")
        f.write("  - `data_loader.py`: 数据加载和预处理\n")
        f.write("  - `lda_classifier.py`: LDA分类器实现\n")
        f.write("  - `decision_tree_classifier.py`: 决策树分类器实现\n")
        f.write("  - `comparison_analysis.py`: 对比分析模块\n")
        f.write("- `main.py`: 主程序入口\n\n")
        
        f.write("### 环境要求\n")
        f.write("```\n")
        f.write("numpy>=1.21.0\n")
        f.write("pandas>=1.3.0\n")
        f.write("scikit-learn>=1.0.0\n")
        f.write("matplotlib>=3.4.0\n")
        f.write("seaborn>=0.11.0\n")
        f.write("```\n")
    
    print(f"综合分析报告已生成: {report_path}")

if __name__ == "__main__":
    main()