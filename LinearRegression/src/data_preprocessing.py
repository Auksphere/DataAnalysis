"""
数据预处理模块
加载和准备波士顿房价数据集
"""
import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_prepare_data():
    """
    加载加州房价数据集（替代波士顿房价数据集）
    返回处理后的数据
    """
    # 加载数据集
    california_housing = fetch_california_housing()
    
    # 创建DataFrame
    feature_names = california_housing.feature_names
    data = pd.DataFrame(california_housing.data, columns=feature_names)
    data['target'] = california_housing.target
    
    print("数据集基本信息：")
    print(f"数据形状: {data.shape}")
    print(f"特征列: {list(feature_names)}")
    print(f"目标变量: 房价中位数（单位：10万美元）")
    
    print("\n数据描述统计：")
    print(data.describe())
    
    print("\n检查缺失值：")
    print(data.isnull().sum())
    
    return data, feature_names

def explore_data(data, feature_names):
    """
    数据探索性分析
    """
    plt.style.use('seaborn-v0_8')
    
    # 创建集成的可视化图表 - 1行3列布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 目标变量分布
    axes[0].hist(data['target'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_title('House Price Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Median House Price (100k USD)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 相关性热力图
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1], fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # 3. 重要特征散点图分析 - 选择与目标变量相关性最高的特征
    correlations = data.corr()['target'].abs().sort_values(ascending=False)
    best_feature = correlations.index[1]  # 排除target本身
    
    scatter = axes[2].scatter(data[best_feature], data['target'], alpha=0.6, 
                             c=data['target'], cmap='viridis', s=20)
    axes[2].set_xlabel(f'{best_feature}')
    axes[2].set_ylabel('Median House Price')
    axes[2].set_title(f'{best_feature} vs House Price', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(scatter, ax=axes[2], label='House Price')
    
    # 调整布局
    plt.suptitle('California Housing Data - Exploratory Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 确保results目录存在
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, 'data_exploration.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    # 不显示图形，只保存
    plt.close()
    
    # 输出相关性分析
    print("\n与目标变量的相关性（绝对值排序）：")
    correlations = data.corr()['target'].abs().sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'target':
            print(f"{feature}: {corr:.3f}")

def preprocess_data(data, feature_names):
    """
    数据预处理
    """
    # 分离特征和目标变量
    X = data[feature_names].copy()
    y = data['target'].copy()
    
    # 移除异常值（使用IQR方法）
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤异常值
    mask = (y >= lower_bound) & (y <= upper_bound)
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"\n移除异常值后的数据形状: {X_clean.shape}")
    print(f"移除了 {len(data) - len(X_clean)} 个异常值")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X_clean.index)
    
    return X_scaled, y_clean, scaler

def save_processed_data(X, y, feature_names):
    """
    保存处理后的数据
    """
    # 合并数据
    processed_data = X.copy()
    processed_data['target'] = y
    
    # 确保data目录存在
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存到CSV文件
    processed_data.to_csv(os.path.join(data_dir, 'processed_data.csv'), 
                         index=False)
    
    # 保存特征名称
    pd.Series(feature_names).to_csv(os.path.join(data_dir, 'feature_names.csv'), 
                                   index=False, header=['feature_name'])
    
    print(f"\n处理后的数据已保存到: data/processed_data.csv")
    print(f"特征名称已保存到: data/feature_names.csv")

if __name__ == "__main__":
    # 加载数据
    data, feature_names = load_and_prepare_data()
    
    # 数据探索
    explore_data(data, feature_names)
    
    # 数据预处理
    X_processed, y_processed, scaler = preprocess_data(data, feature_names)
    
    # 保存处理后的数据
    save_processed_data(X_processed, y_processed, feature_names)
    
    print("\n数据预处理完成！")