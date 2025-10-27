"""
数据加载和预处理模块
使用Wine质量数据集进行分类分析
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import os

class WineDataLoader:
    """Wine数据集加载和预处理类"""
    
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        
    def load_data(self):
        """加载Wine数据集"""
        print("正在加载Wine数据集...")
        
        # 使用sklearn内置的Wine数据集
        wine_data = load_wine()
        
        # 创建DataFrame
        self.df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        self.df['target'] = wine_data.target
        
        self.feature_names = wine_data.feature_names
        self.target_names = wine_data.target_names
        
        print(f"数据集形状: {self.df.shape}")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"类别数量: {len(self.target_names)}")
        print(f"类别分布:")
        print(self.df['target'].value_counts().sort_index())
        
        return self.df
    
    def explore_data(self, save_path='results'):
        """探索性数据分析"""
        print("\n=== 探索性数据分析 ===")
        
        # 确保results目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 基本统计信息
        print("\n数据集基本信息:")
        print(self.df.info())
        
        print("\n数值特征统计:")
        print(self.df.describe())
        
        # 检查缺失值
        missing_values = self.df.isnull().sum()
        print(f"\n缺失值统计:")
        print(missing_values[missing_values > 0])
        
        # 可视化
        self._create_visualizations(save_path)
        
        return self.df.describe()
    
    def _create_visualizations(self, save_path):
        """创建数据可视化"""
        plt.style.use('default')
        
        # 1. 类别分布
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        counts = self.df['target'].value_counts().sort_index()
        bars = plt.bar(range(len(counts)), counts.values)
        plt.xlabel('Wine Class')
        plt.ylabel('Count')
        plt.title('Wine Classes Distribution')
        plt.xticks(range(len(counts)), [f'Class {i}' for i in counts.index])
        
        # 添加数值标签
        for bar, count in zip(bars, counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom')
        
        # 2. 特征相关性热力图
        plt.subplot(2, 2, 2)
        # 选择前10个特征避免图表过于拥挤
        corr_features = self.df.iloc[:, :10].corr()
        sns.heatmap(corr_features, annot=False, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix (Top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 3. 主要特征的分布
        plt.subplot(2, 2, 3)
        for target_class in sorted(self.df['target'].unique()):
            data = self.df[self.df['target'] == target_class]['alcohol']
            plt.hist(data, alpha=0.7, label=f'Class {target_class}', bins=15)
        plt.xlabel('Alcohol Content')
        plt.ylabel('Frequency')
        plt.title('Alcohol Content Distribution by Class')
        plt.legend()
        
        # 4. 箱线图显示特征差异
        plt.subplot(2, 2, 4)
        feature_to_plot = 'flavanoids'  # 选择一个有代表性的特征
        data_to_plot = [self.df[self.df['target'] == i][feature_to_plot].values 
                       for i in sorted(self.df['target'].unique())]
        plt.boxplot(data_to_plot, labels=[f'Class {i}' for i in sorted(self.df['target'].unique())])
        plt.xlabel('Wine Class')
        plt.ylabel('Flavanoids')
        plt.title('Flavanoids Distribution by Class')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'data_exploration.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存特征重要性分析
        self._analyze_feature_importance(save_path)
    
    def _analyze_feature_importance(self, save_path):
        """分析特征重要性"""
        from sklearn.feature_selection import f_classif
        
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # F-统计量分析
        f_stats, p_values = f_classif(X, y)
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Statistic': f_stats,
            'P_Value': p_values
        }).sort_values('F_Statistic', ascending=False)
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        
        bars = plt.barh(range(len(top_features)), top_features['F_Statistic'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('F-Statistic')
        plt.title('Top 10 Most Important Features (F-Statistics)')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存特征重要性到CSV
        feature_importance.to_csv(os.path.join(save_path, 'feature_importance.csv'), index=False)
        
        return feature_importance
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")
        
        # 分离特征和标签
        X = self.df.drop('target', axis=1).values
        y = self.df['target'].values
        
        print(f"原始特征形状: {X.shape}")
        print(f"原始标签形状: {y.shape}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y  # 保持类别比例
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集形状: {X_train_scaled.shape}")
        print(f"测试集形状: {X_test_scaled.shape}")
        print(f"训练集类别分布: {np.bincount(y_train)}")
        print(f"测试集类别分布: {np.bincount(y_test)}")
        
        self.data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_test_raw': X_test
        }
        
        return self.data
    
    def get_data_info(self):
        """获取数据信息"""
        return {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'n_features': len(self.feature_names),
            'n_classes': len(self.target_names),
            'n_samples': len(self.df)
        }
    
    def save_processed_data(self, save_path='data'):
        """保存处理后的数据"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存原始数据
        self.df.to_csv(os.path.join(save_path, 'wine_dataset.csv'), index=False)
        
        # 保存特征名称
        pd.DataFrame({'feature_names': self.feature_names}).to_csv(
            os.path.join(save_path, 'feature_names.csv'), index=False
        )
        
        # 保存目标类别名称
        pd.DataFrame({'target_names': self.target_names}).to_csv(
            os.path.join(save_path, 'target_names.csv'), index=False
        )
        
        print(f"数据已保存到 {save_path} 目录")

if __name__ == "__main__":
    # 测试数据加载器
    loader = WineDataLoader()
    
    # 加载数据
    df = loader.load_data()
    
    # 探索数据
    loader.explore_data()
    
    # 预处理数据
    data = loader.preprocess_data()
    
    # 保存数据
    loader.save_processed_data()
    
    # 显示数据信息
    info = loader.get_data_info()
    print(f"\n数据集信息: {info}")