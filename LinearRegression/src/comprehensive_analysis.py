"""
线性回归综合分析和可视化模块
整合所有分析结果，生成综合报告和可视化
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from .univariate_regression import run_univariate_analysis
from .multivariate_regression import run_multivariate_analysis, compare_univariate_vs_multivariate
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def clean_bootstrap_data(data, name=""):
    """清理bootstrap数据中的异常值"""
    data = np.array(data)
    # 移除无效值（NaN, inf）
    data = data[np.isfinite(data)]
    # 移除极端异常值（使用IQR方法）
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 使用3倍IQR而不是1.5倍，因为bootstrap可能有更大变异
    upper_bound = Q3 + 3 * IQR
    cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    if len(cleaned_data) != len(data):
        print(f"    Warning: {name} Bootstrap数据中移除了 {len(data) - len(cleaned_data)} 个异常值")
    
    return cleaned_data.tolist()

def create_comprehensive_visualization(univariate_results, multivariate_results):
    """
    创建综合可视化图表
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 18))  # 增加高度
    
    # 创建网格布局，增加间距
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)  # 增加垂直间距
    
    # 1. 一元回归散点图和回归线
    ax1 = fig.add_subplot(gs[0, 0])
    holdout_data = univariate_results['holdout_results']
    feature_name = univariate_results['feature']
    
    ax1.scatter(holdout_data['X_train'], holdout_data['y_train'], 
               alpha=0.6, color='blue', label='Training', s=15)
    ax1.scatter(holdout_data['X_val'], holdout_data['y_val'], 
               alpha=0.6, color='red', label='Validation', s=15)
    
    # 回归线
    X_combined = np.concatenate([holdout_data['X_train'], holdout_data['X_val']])
    X_range = np.linspace(X_combined.min(), X_combined.max(), 100).reshape(-1, 1)
    y_range_pred = (holdout_data['model_params']['coef'] * X_range.flatten() + 
                   holdout_data['model_params']['intercept'])
    ax1.plot(X_range, y_range_pred, color='green', linewidth=2, label='Regression Line')
    
    ax1.set_xlabel(f'{feature_name}')
    ax1.set_ylabel('House Price')
    ax1.set_title(f'Univariate Regression: {feature_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 一元回归残差图
    ax2 = fig.add_subplot(gs[0, 1])
    residuals_train = holdout_data['y_train'] - holdout_data['y_train_pred']
    residuals_val = holdout_data['y_val'] - holdout_data['y_val_pred']
    
    ax2.scatter(holdout_data['y_train_pred'], residuals_train, 
               alpha=0.6, color='blue', label='Training', s=15)
    ax2.scatter(holdout_data['y_val_pred'], residuals_val, 
               alpha=0.6, color='red', label='Validation', s=15)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Univariate: Residuals Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 多元回归预测vs实际值
    ax3 = fig.add_subplot(gs[0, 2])
    multi_holdout_data = multivariate_results['holdout_results']
    
    ax3.scatter(multi_holdout_data['y_train'], multi_holdout_data['y_train_pred'], 
               alpha=0.6, color='blue', label='Training', s=15)
    ax3.scatter(multi_holdout_data['y_val'], multi_holdout_data['y_val_pred'], 
               alpha=0.6, color='red', label='Validation', s=15)
    
    # 理想预测线
    min_val = min(np.min(multi_holdout_data['y_train']), np.min(multi_holdout_data['y_val']))
    max_val = max(np.max(multi_holdout_data['y_train']), np.max(multi_holdout_data['y_val']))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
    
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('Multivariate: Predicted vs Actual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 多元回归残差图
    ax4 = fig.add_subplot(gs[0, 3])
    multi_residuals_train = multi_holdout_data['y_train'] - multi_holdout_data['y_train_pred']
    multi_residuals_val = multi_holdout_data['y_val'] - multi_holdout_data['y_val_pred']
    
    ax4.scatter(multi_holdout_data['y_train_pred'], multi_residuals_train, 
               alpha=0.6, color='blue', label='Training', s=15)
    ax4.scatter(multi_holdout_data['y_val_pred'], multi_residuals_val, 
               alpha=0.6, color='red', label='Validation', s=15)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Multivariate: Residuals Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 采样方法比较 - 一元回归
    ax5 = fig.add_subplot(gs[1, 0:2])
    
    methods = ['Hold-out', 'K-Fold CV', 'Bootstrap']
    uni_results = univariate_results['comparison_results']
    
    uni_r2_means = [
        uni_results['holdout']['val_r2'],
        uni_results['kfold']['mean']['val_r2'],
        uni_results['bootstrap']['mean']['val_r2']
    ]
    uni_r2_stds = [
        0,
        uni_results['kfold']['std']['val_r2'],
        uni_results['bootstrap']['std']['val_r2']
    ]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, uni_r2_means, width, yerr=uni_r2_stds, 
                    capsize=5, label='Univariate', color='skyblue', alpha=0.7)
    
    # 采样方法比较 - 多元回归
    multi_results = multivariate_results['comparison_results']
    
    multi_r2_means = [
        multi_results['holdout']['val_r2'],
        multi_results['kfold']['mean']['val_r2'],
        multi_results['bootstrap']['mean']['val_r2']
    ]
    multi_r2_stds = [
        0,
        multi_results['kfold']['std']['val_r2'],
        multi_results['bootstrap']['std']['val_r2']
    ]
    
    bars2 = ax5.bar(x_pos + width/2, multi_r2_means, width, yerr=multi_r2_stds, 
                    capsize=5, label='Multivariate', color='lightcoral', alpha=0.7)
    
    ax5.set_xlabel('Sampling Method')
    ax5.set_ylabel('R² Score')
    ax5.set_title('R² Comparison: Sampling Methods & Model Types')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods, rotation=0, ha='center')  # 确保标签居中，不旋转
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. RMSE比较
    ax6 = fig.add_subplot(gs[1, 2:4])
    
    uni_rmse_means = [
        uni_results['holdout']['val_rmse'],
        uni_results['kfold']['mean']['val_rmse'],
        uni_results['bootstrap']['mean']['val_rmse']
    ]
    uni_rmse_stds = [
        0,
        uni_results['kfold']['std']['val_rmse'],
        uni_results['bootstrap']['std']['val_rmse']
    ]
    
    multi_rmse_means = [
        multi_results['holdout']['val_rmse'],
        multi_results['kfold']['mean']['val_rmse'],
        multi_results['bootstrap']['mean']['val_rmse']
    ]
    multi_rmse_stds = [
        0,
        multi_results['kfold']['std']['val_rmse'],
        multi_results['bootstrap']['std']['val_rmse']
    ]
    
    bars3 = ax6.bar(x_pos - width/2, uni_rmse_means, width, yerr=uni_rmse_stds, 
                    capsize=5, label='Univariate', color='skyblue', alpha=0.7)
    bars4 = ax6.bar(x_pos + width/2, multi_rmse_means, width, yerr=multi_rmse_stds, 
                    capsize=5, label='Multivariate', color='lightcoral', alpha=0.7)
    
    ax6.set_xlabel('Sampling Method')
    ax6.set_ylabel('RMSE')
    ax6.set_title('RMSE Comparison: Sampling Methods & Model Types')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, rotation=0, ha='center')  # 确保标签居中，不旋转
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 特征重要性
    ax7 = fig.add_subplot(gs[2, 0:2])
    feature_importance = multivariate_results['feature_importance']
    
    bars = ax7.barh(feature_importance['feature'], feature_importance['abs_coefficient'], 
                    color='lightgreen', alpha=0.7)
    ax7.set_xlabel('Absolute Coefficient')
    ax7.set_title('Feature Importance (Multivariate Regression)')
    ax7.grid(True, alpha=0.3)
    
    # 8. 模型性能摘要表
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')
    
    # 创建性能摘要
    summary_data = [
        ['Metric', 'Univariate', 'Multivariate', 'Improvement'],
        ['Validation R²', f"{uni_results['holdout']['val_r2']:.4f}", 
         f"{multi_results['holdout']['val_r2']:.4f}", 
         f"+{(multi_results['holdout']['val_r2'] - uni_results['holdout']['val_r2']):.4f}"],
        ['Validation RMSE', f"{uni_results['holdout']['val_rmse']:.4f}", 
         f"{multi_results['holdout']['val_rmse']:.4f}", 
         f"{(multi_results['holdout']['val_rmse'] - uni_results['holdout']['val_rmse']):.4f}"],
        ['Test R²', f"{univariate_results['holdout_scores']['test_r2']:.4f}", 
         f"{multivariate_results['holdout_scores']['test_r2']:.4f}", 
         f"+{(multivariate_results['holdout_scores']['test_r2'] - univariate_results['holdout_scores']['test_r2']):.4f}"],
        ['Test RMSE', f"{univariate_results['holdout_scores']['test_rmse']:.4f}", 
         f"{multivariate_results['holdout_scores']['test_rmse']:.4f}", 
         f"{(multivariate_results['holdout_scores']['test_rmse'] - univariate_results['holdout_scores']['test_rmse']):.4f}"]
    ]
    
    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.8])  # 调整表格位置和大小
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # 减小字体大小
    table.scale(1, 1.5)  # 减小表格缩放
    
    # 设置表格样式，避免重叠
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f1f1f2')
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    ax8.set_title('Performance Summary', pad=10, fontsize=12)  # 减小标题间距
    
    # 9. K折交叉验证结果分布
    ax9 = fig.add_subplot(gs[3, 0:2])
    
    uni_kfold_r2 = [score['val_r2'] for score in univariate_results['kfold_scores']]
    multi_kfold_r2 = [score['val_r2'] for score in multivariate_results['kfold_scores']]
    
    ax9.boxplot([uni_kfold_r2, multi_kfold_r2], labels=['Univariate', 'Multivariate'])
    ax9.set_ylabel('R² Score')
    ax9.set_title('K-Fold Cross-Validation R² Distribution')
    ax9.grid(True, alpha=0.3)
    
        # 10. 自助法结果分布
    ax10 = fig.add_subplot(gs[3, 2:4])
    
    uni_bootstrap_r2 = [score['val_r2'] for score in univariate_results['bootstrap_scores']]
    multi_bootstrap_r2 = [score['val_r2'] for score in multivariate_results['bootstrap_scores']]
    
    # 清理bootstrap数据中的异常值
    uni_bootstrap_r2_clean = clean_bootstrap_data(uni_bootstrap_r2, "Univariate")
    multi_bootstrap_r2_clean = clean_bootstrap_data(multi_bootstrap_r2, "Multivariate")
    
    ax10.boxplot([uni_bootstrap_r2_clean, multi_bootstrap_r2_clean], labels=['Univariate', 'Multivariate'])
    ax10.set_ylabel('R² Score')
    ax10.set_title('Bootstrap R² Distribution')
    ax10.grid(True, alpha=0.3)
    
    plt.suptitle('Linear Regression Analysis: Comprehensive Results', fontsize=16, y=0.95)  # 降低标题位置
    
    # 确保results目录存在
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存图表
    save_path = os.path.join(results_dir, 'comprehensive_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)  # 增加padding
    plt.show()
    
    return save_path

def generate_analysis_summary(univariate_results, multivariate_results):
    """
    生成分析摘要
    """
    summary = {
        'dataset_info': {
            'total_samples': len(univariate_results['holdout_results']['y_train']) + 
                           len(univariate_results['holdout_results']['y_val']) + 
                           len(univariate_results['holdout_results']['y_test']),
            'features_used_univariate': 1,
            'features_used_multivariate': len(multivariate_results['feature_importance']),
            'best_univariate_feature': univariate_results['feature'],
            'univariate_correlation': univariate_results['correlation']
        },
        'performance_comparison': {
            'univariate_val_r2': univariate_results['holdout_scores']['val_r2'],
            'multivariate_val_r2': multivariate_results['holdout_scores']['val_r2'],
            'r2_improvement': multivariate_results['holdout_scores']['val_r2'] - 
                            univariate_results['holdout_scores']['val_r2'],
            'univariate_test_r2': univariate_results['holdout_scores']['test_r2'],
            'multivariate_test_r2': multivariate_results['holdout_scores']['test_r2'],
            'test_r2_improvement': multivariate_results['holdout_scores']['test_r2'] - 
                                 univariate_results['holdout_scores']['test_r2']
        },
        'sampling_method_stability': {
            'univariate_kfold_std': np.std([score['val_r2'] for score in univariate_results['kfold_scores']]),
            'multivariate_kfold_std': np.std([score['val_r2'] for score in multivariate_results['kfold_scores']]),
            'univariate_bootstrap_std': np.std(clean_bootstrap_data([score['val_r2'] for score in univariate_results['bootstrap_scores']], "统计-Univariate")),
            'multivariate_bootstrap_std': np.std(clean_bootstrap_data([score['val_r2'] for score in multivariate_results['bootstrap_scores']], "统计-Multivariate"))
        },
        'feature_importance': multivariate_results['feature_importance'].to_dict('records')
    }
    
    return summary

def print_comprehensive_report(summary):
    """
    打印综合分析报告
    """
    print("=" * 80)
    print("                    线性回归综合分析报告")
    print("=" * 80)
    
    print("\\n1. 数据集信息:")
    print(f"   - 总样本数: {summary['dataset_info']['total_samples']:,}")
    print(f"   - 一元回归使用特征数: {summary['dataset_info']['features_used_univariate']}")
    print(f"   - 多元回归使用特征数: {summary['dataset_info']['features_used_multivariate']}")
    print(f"   - 一元回归最佳特征: {summary['dataset_info']['best_univariate_feature']}")
    print(f"   - 该特征与目标变量相关性: {summary['dataset_info']['univariate_correlation']:.3f}")
    
    print("\\n2. 模型性能比较:")
    perf = summary['performance_comparison']
    print(f"   验证集性能:")
    print(f"   - 一元回归 R²: {perf['univariate_val_r2']:.4f}")
    print(f"   - 多元回归 R²: {perf['multivariate_val_r2']:.4f}")
    print(f"   - R² 提升: +{perf['r2_improvement']:.4f} ({perf['r2_improvement']/perf['univariate_val_r2']*100:.1f}%)")
    
    print(f"   测试集性能:")
    print(f"   - 一元回归 R²: {perf['univariate_test_r2']:.4f}")
    print(f"   - 多元回归 R²: {perf['multivariate_test_r2']:.4f}")
    print(f"   - R² 提升: +{perf['test_r2_improvement']:.4f} ({perf['test_r2_improvement']/perf['univariate_test_r2']*100:.1f}%)")
    
    print("\\n3. 采样方法稳定性分析:")
    stability = summary['sampling_method_stability']
    print(f"   K折交叉验证标准差:")
    print(f"   - 一元回归: {stability['univariate_kfold_std']:.4f}")
    print(f"   - 多元回归: {stability['multivariate_kfold_std']:.4f}")
    
    print(f"   自助法标准差:")
    print(f"   - 一元回归: {stability['univariate_bootstrap_std']:.4f}")
    print(f"   - 多元回归: {stability['multivariate_bootstrap_std']:.4f}")
    
    print("\\n4. 特征重要性排序 (多元回归):")
    for i, feature in enumerate(summary['feature_importance'][:5]):
        print(f"   {i+1}. {feature['feature']}: {feature['abs_coefficient']:.4f}")
    
    print("\\n5. 结论:")
    print(f"   - 多元线性回归相比一元回归在验证集上R²提升了{perf['r2_improvement']:.4f}")
    print(f"   - 在测试集上R²提升了{perf['test_r2_improvement']:.4f}")
    print(f"   - 最重要的特征是 {summary['feature_importance'][0]['feature']}")
    
    if perf['r2_improvement'] > 0.1:
        print("   - 多元回归显著优于一元回归，建议使用多元模型")
    else:
        print("   - 多元回归略优于一元回归")
    
    if stability['multivariate_kfold_std'] < stability['univariate_kfold_std']:
        print("   - 多元回归在K折交叉验证中表现更稳定")
    else:
        print("   - 一元回归在K折交叉验证中表现更稳定")

def run_comprehensive_analysis():
    """
    运行综合分析
    """
    print("正在运行综合线性回归分析...")
    
    # 加载数据
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    data = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
    feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['feature_name'].tolist()
    
    X = data[feature_names]
    y = data['target']
    
    print("\\n运行一元线性回归分析...")
    univariate_results = run_univariate_analysis(X, y, feature_names)
    
    print("\\n运行多元线性回归分析...")
    multivariate_results = run_multivariate_analysis(X, y, feature_names)
    
    print("\\n生成综合可视化...")
    visualization_path = create_comprehensive_visualization(univariate_results, multivariate_results)
    
    print("\\n生成分析摘要...")
    summary = generate_analysis_summary(univariate_results, multivariate_results)
    
    print("\\n生成综合报告...")
    print_comprehensive_report(summary)
    
    print(f"\\n分析完成！综合可视化图表已保存到: {visualization_path}")
    
    return {
        'univariate_results': univariate_results,
        'multivariate_results': multivariate_results,
        'summary': summary,
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    results = run_comprehensive_analysis()