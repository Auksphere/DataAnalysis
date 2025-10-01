"""
数据采样方法模块
实现三种数据采样方法：留出法、K折交叉验证法、自助法
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SamplingMethods:
    """
    数据采样方法类
    """
    
    def __init__(self, X, y, random_state=42):
        """
        初始化采样方法
        
        Parameters:
        X: 特征数据
        y: 目标变量
        random_state: 随机种子
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        np.random.seed(random_state)
        
    def holdout_split(self, test_size=0.2, val_size=0.2):
        """
        留出法（Hold-out）
        将数据分为训练集、验证集和测试集
        
        Parameters:
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集的比例）
        
        Returns:
        训练集、验证集、测试集的特征和标签
        """
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        # 再从剩余数据中分离出训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state
        )
        
        print(f"留出法数据分割:")
        print(f"训练集大小: {len(X_train)} ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"验证集大小: {len(X_val)} ({len(X_val)/len(self.X)*100:.1f}%)")
        print(f"测试集大小: {len(X_test)} ({len(X_test)/len(self.X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def k_fold_split(self, k=5):
        """
        K折交叉验证法
        
        Parameters:
        k: 折数
        
        Returns:
        K折的训练集和验证集索引生成器
        """
        kfold = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
        
        print(f"K折交叉验证 (K={k}):")
        print(f"每折训练集大小: 约 {len(self.X)*(k-1)/k:.0f} ({(k-1)/k*100:.1f}%)")
        print(f"每折验证集大小: 约 {len(self.X)/k:.0f} ({1/k*100:.1f}%)")
        
        return kfold.split(self.X, self.y)
    
    def bootstrap_sample(self, n_samples=None, n_bootstraps=100):
        """
        自助法（Bootstrap）
        
        Parameters:
        n_samples: 每次采样的样本数量，默认为原数据集大小
        n_bootstraps: 自助采样的次数
        
        Returns:
        自助采样的训练集和验证集（out-of-bag样本）
        """
        if n_samples is None:
            n_samples = len(self.X)
        
        bootstrap_results = []
        
        for i in range(n_bootstraps):
            # 有放回采样
            bootstrap_indices = np.random.choice(
                len(self.X), size=n_samples, replace=True
            )
            
            # 获取out-of-bag样本（未被采样的样本）
            oob_indices = np.setdiff1d(np.arange(len(self.X)), bootstrap_indices)
            
            # 构建训练集和验证集
            X_train_bootstrap = self.X.iloc[bootstrap_indices]
            y_train_bootstrap = self.y.iloc[bootstrap_indices]
            
            if len(oob_indices) > 0:
                X_val_bootstrap = self.X.iloc[oob_indices]
                y_val_bootstrap = self.y.iloc[oob_indices]
            else:
                # 如果没有out-of-bag样本，使用一部分训练数据作为验证集
                split_point = int(len(bootstrap_indices) * 0.8)
                train_subset = bootstrap_indices[:split_point]
                val_subset = bootstrap_indices[split_point:]
                
                X_train_bootstrap = self.X.iloc[train_subset]
                y_train_bootstrap = self.y.iloc[train_subset]
                X_val_bootstrap = self.X.iloc[val_subset]
                y_val_bootstrap = self.y.iloc[val_subset]
            
            bootstrap_results.append({
                'X_train': X_train_bootstrap,
                'y_train': y_train_bootstrap,
                'X_val': X_val_bootstrap,
                'y_val': y_val_bootstrap,
                'train_indices': bootstrap_indices,
                'val_indices': oob_indices if len(oob_indices) > 0 else val_subset
            })
        
        print(f"自助法采样:")
        print(f"采样次数: {n_bootstraps}")
        print(f"每次训练集大小: {n_samples}")
        avg_oob_size = np.mean([len(result['val_indices']) for result in bootstrap_results])
        print(f"平均out-of-bag验证集大小: {avg_oob_size:.0f}")
        
        return bootstrap_results

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    评估模型性能
    
    Parameters:
    model: 训练好的模型
    X_train, y_train: 训练集
    X_val, y_val: 验证集
    
    Returns:
    评估指标字典
    """
    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # 验证集预测
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_mae': val_mae
    }

def compare_sampling_methods_performance(holdout_scores, kfold_scores, bootstrap_scores):
    """
    比较三种采样方法的性能
    
    Parameters:
    holdout_scores: 留出法评估结果
    kfold_scores: K折交叉验证评估结果列表
    bootstrap_scores: 自助法评估结果列表
    
    Returns:
    比较结果的汇总
    """
    # 计算K折和自助法的平均性能
    kfold_avg = {
        metric: np.mean([score[metric] for score in kfold_scores])
        for metric in kfold_scores[0].keys()
    }
    kfold_std = {
        metric: np.std([score[metric] for score in kfold_scores])
        for metric in kfold_scores[0].keys()
    }
    
    bootstrap_avg = {
        metric: np.mean([score[metric] for score in bootstrap_scores])
        for metric in bootstrap_scores[0].keys()
    }
    bootstrap_std = {
        metric: np.std([score[metric] for score in bootstrap_scores])
        for metric in bootstrap_scores[0].keys()
    }
    
    # 创建比较表格
    comparison_data = {
        'Sampling Method': ['Hold-out', 'K-Fold CV', 'Bootstrap'],
        'Val RMSE (Mean±Std)': [
            f"{holdout_scores['val_rmse']:.4f}",
            f"{kfold_avg['val_rmse']:.4f}±{kfold_std['val_rmse']:.4f}",
            f"{bootstrap_avg['val_rmse']:.4f}±{bootstrap_std['val_rmse']:.4f}"
        ],
        'Val R² (Mean±Std)': [
            f"{holdout_scores['val_r2']:.4f}",
            f"{kfold_avg['val_r2']:.4f}±{kfold_std['val_r2']:.4f}",
            f"{bootstrap_avg['val_r2']:.4f}±{bootstrap_std['val_r2']:.4f}"
        ],
        'Val MAE (Mean±Std)': [
            f"{holdout_scores['val_mae']:.4f}",
            f"{kfold_avg['val_mae']:.4f}±{kfold_std['val_mae']:.4f}",
            f"{bootstrap_avg['val_mae']:.4f}±{bootstrap_std['val_mae']:.4f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df, {
        'holdout': holdout_scores,
        'kfold': {'mean': kfold_avg, 'std': kfold_std, 'all_scores': kfold_scores},
        'bootstrap': {'mean': bootstrap_avg, 'std': bootstrap_std, 'all_scores': bootstrap_scores}
    }

def visualize_sampling_comparison(comparison_results, model_name="Linear Regression"):
    """
    可视化采样方法比较结果
    """
    plt.style.use('default')  # 使用默认样式避免中文显示问题
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取数据
    methods = ['Hold-out', 'K-Fold CV', 'Bootstrap']
    
    # 验证集RMSE比较
    holdout_rmse = comparison_results['holdout']['val_rmse']
    kfold_rmse_mean = comparison_results['kfold']['mean']['val_rmse']
    kfold_rmse_std = comparison_results['kfold']['std']['val_rmse']
    bootstrap_rmse_mean = comparison_results['bootstrap']['mean']['val_rmse']
    bootstrap_rmse_std = comparison_results['bootstrap']['std']['val_rmse']
    
    rmse_means = [holdout_rmse, kfold_rmse_mean, bootstrap_rmse_mean]
    rmse_stds = [0, kfold_rmse_std, bootstrap_rmse_std]
    
    axes[0, 0].bar(methods, rmse_means, yerr=rmse_stds, capsize=5, 
                   color=['skyblue', 'lightgreen', 'salmon'], alpha=0.7)
    axes[0, 0].set_title(f'Validation RMSE Comparison - {model_name}')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 验证集R²比较
    holdout_r2 = comparison_results['holdout']['val_r2']
    kfold_r2_mean = comparison_results['kfold']['mean']['val_r2']
    kfold_r2_std = comparison_results['kfold']['std']['val_r2']
    bootstrap_r2_mean = comparison_results['bootstrap']['mean']['val_r2']
    bootstrap_r2_std = comparison_results['bootstrap']['std']['val_r2']
    
    r2_means = [holdout_r2, kfold_r2_mean, bootstrap_r2_mean]
    r2_stds = [0, kfold_r2_std, bootstrap_r2_std]
    
    axes[0, 1].bar(methods, r2_means, yerr=r2_stds, capsize=5,
                   color=['skyblue', 'lightgreen', 'salmon'], alpha=0.7)
    axes[0, 1].set_title(f'Validation R² Comparison - {model_name}')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # K折交叉验证结果分布
    kfold_rmse_scores = [score['val_rmse'] for score in comparison_results['kfold']['all_scores']]
    axes[1, 0].boxplot([kfold_rmse_scores], labels=['K-Fold CV'])
    axes[1, 0].set_title('K-Fold CV RMSE Distribution')
    axes[1, 0].set_ylabel('RMSE')
    
    # 自助法结果分布
    bootstrap_rmse_scores = [score['val_rmse'] for score in comparison_results['bootstrap']['all_scores']]
    axes[1, 1].boxplot([bootstrap_rmse_scores], labels=['Bootstrap'])
    axes[1, 1].set_title('Bootstrap RMSE Distribution')
    axes[1, 1].set_ylabel('RMSE')
    
    plt.tight_layout()
    
    # 确保results目录存在
    project_root = get_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存图表
    filename = os.path.join(results_dir, f'sampling_comparison_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 不显示图形，只保存
    
    return filename

if __name__ == "__main__":
    # 加载处理后的数据
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    data = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
    feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['feature_name'].tolist()
    
    X = data[feature_names]
    y = data['target']
    
    print("=== 数据采样方法比较 ===\n")
    
    # 初始化采样方法
    sampler = SamplingMethods(X, y)
    
    # 测试留出法
    print("1. 留出法测试:")
    X_train_hold, X_val_hold, X_test_hold, y_train_hold, y_val_hold, y_test_hold = sampler.holdout_split()
    print()
    
    # 测试K折交叉验证法
    print("2. K折交叉验证法测试:")
    kfold_splits = list(sampler.k_fold_split(k=5))
    print(f"生成了 {len(kfold_splits)} 个训练-验证集对")
    print()
    
    # 测试自助法
    print("3. 自助法测试:")
    bootstrap_samples = sampler.bootstrap_sample(n_bootstraps=50)
    print(f"生成了 {len(bootstrap_samples)} 个自助采样")
    print()
    
    print("数据采样方法实现完成！")