"""
一元线性回归分析模块
使用三种采样方法进行一元线性回归分析
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from .sampling_methods import SamplingMethods, evaluate_model, compare_sampling_methods_performance, visualize_sampling_comparison
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UnivariateLinearRegression:
    """
    一元线性回归分析类
    """
    
    def __init__(self, X, y, feature_name):
        """
        初始化一元线性回归分析
        
        Parameters:
        X: 特征数据（单个特征）
        y: 目标变量
        feature_name: 特征名称
        """
        self.X = X.values.reshape(-1, 1) if hasattr(X, 'values') else X.reshape(-1, 1)
        self.y = y
        self.feature_name = feature_name
        self.model = LinearRegression()
        
    def fit_and_evaluate_holdout(self, sampler):
        """
        使用留出法训练和评估模型
        """
        # 获取留出法分割的数据
        X_train, X_val, X_test, y_train, y_val, y_test = sampler.holdout_split()
        
        # 选择单个特征进行一元回归
        X_train_uni = X_train[self.feature_name].values.reshape(-1, 1)
        X_val_uni = X_val[self.feature_name].values.reshape(-1, 1)
        X_test_uni = X_test[self.feature_name].values.reshape(-1, 1)
        
        # 训练模型
        self.model.fit(X_train_uni, y_train)
        
        # 评估模型
        scores = evaluate_model(self.model, X_train_uni, y_train, X_val_uni, y_val)
        
        # 在测试集上的性能
        y_test_pred = self.model.predict(X_test_uni)
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
        
        # 保存预测结果用于可视化
        self.holdout_results = {
            'X_train': X_train_uni,
            'y_train': y_train,
            'X_val': X_val_uni,
            'y_val': y_val,
            'X_test': X_test_uni,
            'y_test': y_test,
            'y_train_pred': self.model.predict(X_train_uni),
            'y_val_pred': self.model.predict(X_val_uni),
            'y_test_pred': y_test_pred,
            'model_params': {
                'coef': self.model.coef_[0],
                'intercept': self.model.intercept_
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
            
            # 选择单个特征
            X_train_uni = X_fold[self.feature_name].values.reshape(-1, 1)
            X_val_uni = X_val_fold[self.feature_name].values.reshape(-1, 1)
            
            # 训练模型
            fold_model = LinearRegression()
            fold_model.fit(X_train_uni, y_fold)
            
            # 评估模型
            scores = evaluate_model(fold_model, X_train_uni, y_fold, X_val_uni, y_val_fold)
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
            
            # 选择单个特征
            X_train_uni = X_train_bootstrap[self.feature_name].values.reshape(-1, 1)
            X_val_uni = X_val_bootstrap[self.feature_name].values.reshape(-1, 1)
            
            # 训练模型
            bootstrap_model = LinearRegression()
            bootstrap_model.fit(X_train_uni, y_train_bootstrap)
            
            # 评估模型
            scores = evaluate_model(bootstrap_model, X_train_uni, y_train_bootstrap, X_val_uni, y_val_bootstrap)
            bootstrap_scores.append(scores)
            
            if (i+1) % 10 == 0:
                print(f"  Bootstrap {i+1}: Val R² = {scores['val_r2']:.4f}, Val RMSE = {scores['val_rmse']:.4f}")
        
        return bootstrap_scores

def run_univariate_analysis(X, y, feature_names):
    """
    对所有特征进行一元线性回归分析
    """
    # 选择与目标变量相关性最高的特征进行详细分析
    correlations = {}
    for feature in feature_names:
        corr = np.corrcoef(X[feature], y)[0, 1]
        correlations[feature] = abs(corr)
    
    # 选择相关性最高的特征
    best_feature = max(correlations, key=correlations.get)
    
    print(f"=== 一元线性回归分析 ===")
    print(f"选择特征: {best_feature} (相关性: {correlations[best_feature]:.3f})")
    
    # 初始化采样方法和一元回归模型
    sampler = SamplingMethods(X, y)
    univariate_model = UnivariateLinearRegression(X[best_feature], y, best_feature)
    
    # 1. 留出法
    print("\\n1. 留出法分析:")
    holdout_scores = univariate_model.fit_and_evaluate_holdout(sampler)
    print(f"  验证集 R²: {holdout_scores['val_r2']:.4f}")
    print(f"  验证集 RMSE: {holdout_scores['val_rmse']:.4f}")
    print(f"  测试集 R²: {holdout_scores['test_r2']:.4f}")
    print(f"  测试集 RMSE: {holdout_scores['test_rmse']:.4f}")
    
    # 2. K折交叉验证
    print("\\n2. K折交叉验证分析:")
    kfold_scores = univariate_model.fit_and_evaluate_kfold(sampler, k=5)
    kfold_mean_r2 = np.mean([score['val_r2'] for score in kfold_scores])
    kfold_std_r2 = np.std([score['val_r2'] for score in kfold_scores])
    kfold_mean_rmse = np.mean([score['val_rmse'] for score in kfold_scores])
    kfold_std_rmse = np.std([score['val_rmse'] for score in kfold_scores])
    print(f"  平均验证集 R²: {kfold_mean_r2:.4f} ± {kfold_std_r2:.4f}")
    print(f"  平均验证集 RMSE: {kfold_mean_rmse:.4f} ± {kfold_std_rmse:.4f}")
    
    # 3. 自助法
    print("\\n3. 自助法分析:")
    bootstrap_scores = univariate_model.fit_and_evaluate_bootstrap(sampler, n_bootstraps=50)
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
    
    # 保存详细结果
    results_summary = {
        'feature': best_feature,
        'correlation': correlations[best_feature],
        'holdout_scores': holdout_scores,
        'kfold_scores': kfold_scores,
        'bootstrap_scores': bootstrap_scores,
        'comparison_df': comparison_df,
        'comparison_results': comparison_results,
        'holdout_results': univariate_model.holdout_results
    }
    
    return results_summary

if __name__ == "__main__":
    # 加载数据
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    data = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
    feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['feature_name'].tolist()
    
    X = data[feature_names]
    y = data['target']
    
    # 运行一元线性回归分析
    results = run_univariate_analysis(X, y, feature_names)
    
    print("\\n一元线性回归分析完成！")
    print(f"分析的特征: {results['feature']}")
    print(f"特征相关性: {results['correlation']:.3f}")