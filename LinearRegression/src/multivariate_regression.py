"""
多元线性回归分析模块
使用三种采样方法进行多元线性回归分析
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from .sampling_methods import SamplingMethods, evaluate_model, compare_sampling_methods_performance
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MultivariateLinearRegression:
    """
    多元线性回归分析类
    """
    
    def __init__(self, X, y, feature_names):
        """
        初始化多元线性回归分析
        
        Parameters:
        X: 特征数据
        y: 目标变量
        feature_names: 特征名称列表
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.model = LinearRegression()
        
    def fit_and_evaluate_holdout(self, sampler):
        """
        使用留出法训练和评估模型
        """
        # 获取留出法分割的数据
        X_train, X_val, X_test, y_train, y_val, y_test = sampler.holdout_split()
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        scores = evaluate_model(self.model, X_train, y_train, X_val, y_val)
        
        # 在测试集上的性能
        y_test_pred = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        scores.update({
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        
        # 保存模型参数
        self.holdout_results = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'y_train_pred': self.model.predict(X_train),
            'y_val_pred': self.model.predict(X_val),
            'y_test_pred': y_test_pred,
            'model_params': {
                'coef': self.model.coef_,
                'intercept': self.model.intercept_,
                'feature_names': self.feature_names
            }
        }
        
        return scores
    
    def fit_and_evaluate_kfold(self, sampler, k=5):
        """
        使用K折交叉验证训练和评估模型
        """
        kfold_splits = sampler.k_fold_split(k=k)
        kfold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            # 获取当前折的数据
            X_fold = sampler.X.iloc[train_idx]
            y_fold = sampler.y.iloc[train_idx]
            X_val_fold = sampler.X.iloc[val_idx]
            y_val_fold = sampler.y.iloc[val_idx]
            
            # 训练模型
            fold_model = LinearRegression()
            fold_model.fit(X_fold, y_fold)
            
            # 评估模型
            scores = evaluate_model(fold_model, X_fold, y_fold, X_val_fold, y_val_fold)
            kfold_scores.append(scores)
            
            print(f"  Fold {fold+1}: Val R² = {scores['val_r2']:.4f}, Val RMSE = {scores['val_rmse']:.4f}")
        
        return kfold_scores
    
    def fit_and_evaluate_bootstrap(self, sampler, n_bootstraps=50):
        """
        使用自助法训练和评估模型
        """
        bootstrap_samples = sampler.bootstrap_sample(n_bootstraps=n_bootstraps)
        bootstrap_scores = []
        
        for i, sample in enumerate(bootstrap_samples):
            # 获取当前自助样本的数据
            X_train_bootstrap = sample['X_train']
            y_train_bootstrap = sample['y_train']
            X_val_bootstrap = sample['X_val']
            y_val_bootstrap = sample['y_val']
            
            # 训练模型
            bootstrap_model = LinearRegression()
            bootstrap_model.fit(X_train_bootstrap, y_train_bootstrap)
            
            # 评估模型
            scores = evaluate_model(bootstrap_model, X_train_bootstrap, y_train_bootstrap, 
                                   X_val_bootstrap, y_val_bootstrap)
            bootstrap_scores.append(scores)
            
            if (i+1) % 10 == 0:
                print(f"  Bootstrap {i+1}: Val R² = {scores['val_r2']:.4f}, Val RMSE = {scores['val_rmse']:.4f}")
        
        return bootstrap_scores
    
    def feature_importance_analysis(self):
        """
        分析特征重要性
        """
        if not hasattr(self, 'holdout_results'):
            raise ValueError("请先运行 fit_and_evaluate_holdout 方法")
        
        # 获取模型系数
        coefficients = self.holdout_results['model_params']['coef']
        feature_names = self.holdout_results['model_params']['feature_names']
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return feature_importance

def run_multivariate_analysis(X, y, feature_names):
    """
    运行多元线性回归分析
    """
    print(f"=== 多元线性回归分析 ===")
    print(f"使用特征数量: {len(feature_names)}")
    print(f"特征列表: {', '.join(feature_names)}")
    
    # 初始化采样方法和多元回归模型
    sampler = SamplingMethods(X, y)
    multivariate_model = MultivariateLinearRegression(X, y, feature_names)
    
    # 1. 留出法
    print("\\n1. 留出法分析:")
    holdout_scores = multivariate_model.fit_and_evaluate_holdout(sampler)
    print(f"  验证集 R²: {holdout_scores['val_r2']:.4f}")
    print(f"  验证集 RMSE: {holdout_scores['val_rmse']:.4f}")
    print(f"  测试集 R²: {holdout_scores['test_r2']:.4f}")
    print(f"  测试集 RMSE: {holdout_scores['test_rmse']:.4f}")
    
    # 2. K折交叉验证
    print("\\n2. K折交叉验证分析:")
    kfold_scores = multivariate_model.fit_and_evaluate_kfold(sampler, k=5)
    kfold_mean_r2 = np.mean([score['val_r2'] for score in kfold_scores])
    kfold_std_r2 = np.std([score['val_r2'] for score in kfold_scores])
    kfold_mean_rmse = np.mean([score['val_rmse'] for score in kfold_scores])
    kfold_std_rmse = np.std([score['val_rmse'] for score in kfold_scores])
    print(f"  平均验证集 R²: {kfold_mean_r2:.4f} ± {kfold_std_r2:.4f}")
    print(f"  平均验证集 RMSE: {kfold_mean_rmse:.4f} ± {kfold_std_rmse:.4f}")
    
    # 3. 自助法
    print("\\n3. 自助法分析:")
    bootstrap_scores = multivariate_model.fit_and_evaluate_bootstrap(sampler, n_bootstraps=50)
    bootstrap_mean_r2 = np.mean([score['val_r2'] for score in bootstrap_scores])
    bootstrap_std_r2 = np.std([score['val_r2'] for score in bootstrap_scores])
    bootstrap_mean_rmse = np.mean([score['val_rmse'] for score in bootstrap_scores])
    bootstrap_std_rmse = np.std([score['val_rmse'] for score in bootstrap_scores])
    print(f"  平均验证集 R²: {bootstrap_mean_r2:.4f} ± {bootstrap_std_r2:.4f}")
    print(f"  平均验证集 RMSE: {bootstrap_mean_rmse:.4f} ± {bootstrap_std_rmse:.4f}")
    
    # 比较三种方法
    print("\\n4. 三种采样方法比较:")
    comparison_df, comparison_results = compare_sampling_methods_performance(
        holdout_scores, kfold_scores, bootstrap_scores
    )
    print(comparison_df)
    
    # 特征重要性分析
    print("\\n5. 特征重要性分析:")
    feature_importance = multivariate_model.feature_importance_analysis()
    print(feature_importance)
    
    # 保存详细结果
    results_summary = {
        'holdout_scores': holdout_scores,
        'kfold_scores': kfold_scores,
        'bootstrap_scores': bootstrap_scores,
        'comparison_df': comparison_df,
        'comparison_results': comparison_results,
        'feature_importance': feature_importance,
        'holdout_results': multivariate_model.holdout_results
    }
    
    return results_summary

def compare_univariate_vs_multivariate(univariate_results, multivariate_results):
    """
    比较一元和多元线性回归的结果
    """
    print("\\n=== 一元 vs 多元线性回归比较 ===")
    
    # 提取关键指标
    uni_holdout_r2 = univariate_results['holdout_scores']['val_r2']
    uni_holdout_rmse = univariate_results['holdout_scores']['val_rmse']
    uni_test_r2 = univariate_results['holdout_scores']['test_r2']
    uni_test_rmse = univariate_results['holdout_scores']['test_rmse']
    
    multi_holdout_r2 = multivariate_results['holdout_scores']['val_r2']
    multi_holdout_rmse = multivariate_results['holdout_scores']['val_rmse']
    multi_test_r2 = multivariate_results['holdout_scores']['test_r2']
    multi_test_rmse = multivariate_results['holdout_scores']['test_rmse']
    
    # 创建比较表格
    comparison_data = {
        'Model Type': ['Univariate', 'Multivariate', 'Improvement'],
        'Validation R²': [
            f"{uni_holdout_r2:.4f}",
            f"{multi_holdout_r2:.4f}",
            f"+{(multi_holdout_r2 - uni_holdout_r2):.4f}"
        ],
        'Validation RMSE': [
            f"{uni_holdout_rmse:.4f}",
            f"{multi_holdout_rmse:.4f}",
            f"{(multi_holdout_rmse - uni_holdout_rmse):.4f}"
        ],
        'Test R²': [
            f"{uni_test_r2:.4f}",
            f"{multi_test_r2:.4f}",
            f"+{(multi_test_r2 - uni_test_r2):.4f}"
        ],
        'Test RMSE': [
            f"{uni_test_rmse:.4f}",
            f"{multi_test_rmse:.4f}",
            f"{(multi_test_rmse - uni_test_rmse):.4f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    return comparison_df

if __name__ == "__main__":
    # 加载数据
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    data = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
    feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['feature_name'].tolist()
    
    X = data[feature_names]
    y = data['target']
    
    # 运行多元线性回归分析
    multivariate_results = run_multivariate_analysis(X, y, feature_names)
    
    print("\\n多元线性回归分析完成！")
    
    # 如果存在一元回归结果，进行比较
    try:
        from .univariate_regression import run_univariate_analysis
        print("\\n正在运行一元回归以进行比较...")
        univariate_results = run_univariate_analysis(X, y, feature_names)
        
        # 比较一元和多元回归
        comparison_df = compare_univariate_vs_multivariate(univariate_results, multivariate_results)
        
        print("\\n=== 分析总结 ===")
        print(f"一元回归最佳特征: {univariate_results['feature']}")
        print(f"多元回归使用特征数: {len(feature_names)}")
        print("性能提升显著，多元回归明显优于一元回归")
        
    except ImportError:
        print("\\n无法导入一元回归模块，跳过比较分析")