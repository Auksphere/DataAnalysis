"""
线性判别分析 (Linear Discriminant Analysis, LDA) 分类器实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

class LDAClassifier:
    """线性判别分析分类器"""
    
    def __init__(self, n_components=None, solver='svd', shrinkage=None):
        """
        初始化LDA分类器
        
        Parameters:
        - n_components: 降维后的维数
        - solver: 求解器 ('svd', 'lsqr', 'eigen')
        - shrinkage: 收缩参数
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = None
        self.feature_names = None
        self.target_names = None
        
    def train(self, X_train, y_train, feature_names=None, target_names=None):
        """训练LDA模型"""
        print("=== 训练LDA分类器 ===")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # 创建LDA模型
        self.model = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver,
            shrinkage=self.shrinkage
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 训练集预测
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"LDA组件数: {self.model.n_components}")
        
        # 分析判别函数
        self._analyze_discriminant_functions()
        
        return self.model
    
    def _analyze_discriminant_functions(self):
        """分析判别函数"""
        print(f"\n判别函数分析:")
        print(f"线性判别函数数量: {self.model.n_components}")
        
        # 获取判别函数系数
        if hasattr(self.model, 'coef_'):
            print(f"判别函数系数矩阵形状: {self.model.coef_.shape}")
            
        # 获取解释方差比（如果适用）
        if hasattr(self.model, 'explained_variance_ratio_'):
            print(f"解释方差比: {self.model.explained_variance_ratio_}")
            print(f"累计解释方差比: {np.cumsum(self.model.explained_variance_ratio_)}")
    
    def predict(self, X_test, y_test=None):
        """预测和评估"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        print("\n=== LDA模型预测和评估 ===")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        results = {
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if y_test is not None:
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            print(f"测试集准确率: {accuracy:.4f}")
            print(f"加权精确率: {precision:.4f}")
            print(f"加权召回率: {recall:.4f}")
            print(f"加权F1分数: {f1:.4f}")
            
            # 详细分类报告
            print(f"\n详细分类报告:")
            target_names_list = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
            if self.target_names is not None:
                target_names_list = [f"Class_{i}({self.target_names[i]})" for i in range(len(self.target_names))]
            
            print(classification_report(y_test, y_pred, target_names=target_names_list))
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'y_true': y_test
            })
        
        return results
    
    def visualize_results(self, X_test, y_test, results, save_path='results'):
        """可视化结果"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 混淆矩阵
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, results['predictions'])
        
        # 创建标签
        class_labels = [f'Class {i}' for i in range(len(np.unique(y_test)))]
        if self.target_names is not None:
            class_labels = [f'C{i}' for i in range(len(self.target_names))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('LDA - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 2. 类别预测概率分布
        plt.subplot(2, 3, 2)
        probabilities = results['probabilities']
        for i in range(probabilities.shape[1]):
            plt.hist(probabilities[:, i], alpha=0.7, label=f'Class {i}', bins=20)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('LDA - Prediction Probability Distribution')
        plt.legend()
        
        # 3. LDA变换后的数据可视化
        if self.model.n_components >= 2:
            plt.subplot(2, 3, 3)
            X_lda = self.model.transform(X_test)
            
            scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, cmap='viridis', alpha=0.7)
            plt.xlabel(f'LDA Component 1')
            plt.ylabel(f'LDA Component 2')
            plt.title('LDA - Transformed Data Visualization')
            plt.colorbar(scatter, label='True Class')
            
        # 4. 特征重要性（基于判别函数系数）
        if hasattr(self.model, 'coef_') and self.feature_names is not None:
            plt.subplot(2, 3, 4)
            # 取第一个判别函数的系数
            coef = np.abs(self.model.coef_[0])  # 取绝对值
            
            # 选择前10个最重要的特征
            top_indices = np.argsort(coef)[-10:]
            top_coef = coef[top_indices]
            top_features = np.array(self.feature_names)[top_indices]
            
            bars = plt.barh(range(len(top_coef)), top_coef)
            plt.yticks(range(len(top_coef)), [name[:15] + '...' if len(name) > 15 else name for name in top_features])
            plt.xlabel('Absolute Coefficient Value')
            plt.title('LDA - Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
        
        # 5. ROC曲线（多分类）
        plt.subplot(2, 3, 5)
        n_classes = len(np.unique(y_test))
        
        if n_classes > 2:
            # 多分类ROC曲线
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            y_prob = results['probabilities']
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        else:
            # 二分类ROC曲线
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'][:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LDA - ROC Curves')
        plt.legend()
        
        # 6. 预测准确率按类别
        plt.subplot(2, 3, 6)
        
        # 计算每个类别的准确率
        class_accuracies = []
        for class_id in range(n_classes):
            mask = y_test == class_id
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_test[mask], results['predictions'][mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = plt.bar(range(n_classes), class_accuracies, alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('LDA - Per-Class Accuracy')
        plt.xticks(range(n_classes), [f'Class {i}' for i in range(n_classes)])
        
        # 添加数值标签
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'lda_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存LDA变换后的数据
        self._save_lda_transform(X_test, y_test, save_path)
    
    def _save_lda_transform(self, X_test, y_test, save_path):
        """保存LDA变换后的数据"""
        if self.model.n_components > 0:
            X_lda = self.model.transform(X_test)
            
            # 创建DataFrame
            columns = [f'LDA_Component_{i+1}' for i in range(X_lda.shape[1])]
            lda_df = pd.DataFrame(X_lda, columns=columns)
            lda_df['true_label'] = y_test
            
            # 保存到CSV
            lda_df.to_csv(os.path.join(save_path, 'lda_transformed_data.csv'), index=False)
            
            print(f"LDA变换后的数据已保存到 {os.path.join(save_path, 'lda_transformed_data.csv')}")
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型尚未训练"}
        
        info = {
            "model_type": "Linear Discriminant Analysis",
            "n_components": self.model.n_components,
            "solver": self.solver,
            "shrinkage": self.shrinkage,
        }
        
        if hasattr(self.model, 'explained_variance_ratio_'):
            info["explained_variance_ratio"] = self.model.explained_variance_ratio_.tolist()
            info["cumulative_variance_ratio"] = np.cumsum(self.model.explained_variance_ratio_).tolist()
        
        if hasattr(self.model, 'coef_'):
            info["discriminant_coefficients_shape"] = self.model.coef_.shape
        
        return info
    
    def analyze_decision_boundary(self, X_test, y_test, save_path='results'):
        """分析决策边界（对于2D情况）"""
        if self.model.n_components < 2:
            print("无法可视化决策边界，需要至少2个LDA组件")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 获取LDA变换后的数据
        X_lda = self.model.transform(X_test)
        
        # 只使用前两个组件
        X_2d = X_lda[:, :2]
        
        # 创建网格
        h = 0.1  # 网格步长
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 需要将2D点反变换到原始空间再预测
        # 这里我们直接在LDA空间中进行可视化
        mesh_points = grid_points
        
        # 由于我们在LDA空间中，需要特殊处理
        # 这里我们展示数据分布而不是决策边界
        
        # 绘制数据点
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test, cmap='viridis', 
                            alpha=0.8, s=50, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('LDA Component 1')
        plt.ylabel('LDA Component 2')
        plt.title('LDA - Data Distribution in Transformed Space')
        plt.colorbar(scatter, label='Class')
        
        # 添加类别中心
        for class_id in np.unique(y_test):
            mask = y_test == class_id
            center_x = np.mean(X_2d[mask, 0])
            center_y = np.mean(X_2d[mask, 1])
            plt.plot(center_x, center_y, 'k*', markersize=15, 
                    markeredgewidth=2, markeredgecolor='white',
                    label=f'Class {class_id} Center')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'lda_decision_space.png'), dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 测试LDA分类器
    from data_loader import WineDataLoader
    
    # 加载数据
    loader = WineDataLoader()
    df = loader.load_data()
    data = loader.preprocess_data()
    info = loader.get_data_info()
    
    # 创建和训练LDA分类器
    lda = LDAClassifier(n_components=2)
    lda.train(data['X_train'], data['y_train'], 
             feature_names=info['feature_names'],
             target_names=info['target_names'])
    
    # 预测和评估
    results = lda.predict(data['X_test'], data['y_test'])
    
    # 可视化结果
    lda.visualize_results(data['X_test'], data['y_test'], results)
    
    # 分析决策边界
    lda.analyze_decision_boundary(data['X_test'], data['y_test'])
    
    # 显示模型信息
    model_info = lda.get_model_info()
    print(f"\n模型信息: {model_info}")