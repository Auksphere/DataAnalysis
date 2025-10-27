"""
决策树 (Decision Tree) 分类器实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeClassifierCustom:
    """决策树分类器"""
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, random_state=42):
        """
        初始化决策树分类器
        
        Parameters:
        - criterion: 分割标准 ('gini', 'entropy')
        - max_depth: 最大深度
        - min_samples_split: 分割所需最小样本数
        - min_samples_leaf: 叶节点最小样本数
        - max_features: 最大特征数
        - random_state: 随机种子
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.target_names = None
        
    def train(self, X_train, y_train, feature_names=None, target_names=None):
        """训练决策树模型"""
        print("=== 训练决策树分类器 ===")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # 创建决策树模型
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 训练集预测
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"决策树深度: {self.model.get_depth()}")
        print(f"叶节点数量: {self.model.get_n_leaves()}")
        
        # 分析决策树结构
        self._analyze_tree_structure()
        
        return self.model
    
    def _analyze_tree_structure(self):
        """分析决策树结构"""
        print(f"\n决策树结构分析:")
        print(f"树的深度: {self.model.get_depth()}")
        print(f"叶节点数量: {self.model.get_n_leaves()}")
        print(f"节点总数: {self.model.tree_.node_count}")
        
        # 获取特征重要性
        if self.feature_names is not None:
            feature_importance = self.model.feature_importances_
            important_features = [(self.feature_names[i], importance) 
                                for i, importance in enumerate(feature_importance) if importance > 0.01]
            important_features.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n重要特征 (重要性 > 0.01):")
            for feature, importance in important_features[:10]:
                print(f"  {feature}: {importance:.4f}")
    
    def predict(self, X_test, y_test=None):
        """预测和评估"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        print("\n=== 决策树模型预测和评估 ===")
        
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
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 混淆矩阵
        plt.subplot(2, 4, 1)
        cm = confusion_matrix(y_test, results['predictions'])
        
        # 创建标签
        class_labels = [f'Class {i}' for i in range(len(np.unique(y_test)))]
        if self.target_names is not None:
            class_labels = [f'C{i}' for i in range(len(self.target_names))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Decision Tree - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 2. 类别预测概率分布
        plt.subplot(2, 4, 2)
        probabilities = results['probabilities']
        for i in range(probabilities.shape[1]):
            plt.hist(probabilities[:, i], alpha=0.7, label=f'Class {i}', bins=20)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Decision Tree - Prediction Probability Distribution')
        plt.legend()
        
        # 3. 特征重要性
        plt.subplot(2, 4, 3)
        if self.feature_names is not None:
            feature_importance = self.model.feature_importances_
            
            # 选择前10个最重要的特征
            top_indices = np.argsort(feature_importance)[-10:]
            top_importance = feature_importance[top_indices]
            top_features = np.array(self.feature_names)[top_indices]
            
            bars = plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(top_importance)), 
                      [name[:15] + '...' if len(name) > 15 else name for name in top_features])
            plt.xlabel('Feature Importance')
            plt.title('Decision Tree - Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
        
        # 4. ROC曲线（多分类）
        plt.subplot(2, 4, 4)
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
        plt.title('Decision Tree - ROC Curves')
        plt.legend()
        
        # 5. 预测准确率按类别
        plt.subplot(2, 4, 5)
        
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
        plt.title('Decision Tree - Per-Class Accuracy')
        plt.xticks(range(n_classes), [f'Class {i}' for i in range(n_classes)])
        
        # 添加数值标签
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 6. 决策树可视化（简化版）
        plt.subplot(2, 4, 6)
        if self.model.get_depth() <= 4:  # 只在深度较小时可视化
            plot_tree(self.model, max_depth=3, filled=True, 
                     feature_names=[name[:8] for name in self.feature_names] if self.feature_names else None,
                     class_names=[f'C{i}' for i in range(n_classes)],
                     fontsize=8)
            plt.title('Decision Tree Structure (Max Depth=3)')
        else:
            plt.text(0.5, 0.5, f'Tree too complex\nto visualize\n(Depth: {self.model.get_depth()})', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Decision Tree - Too Complex to Visualize')
        plt.axis('off')
        
        # 7. 树的深度分析
        plt.subplot(2, 4, 7)
        # 模拟不同深度的性能（这里简化处理）
        depths = range(1, min(11, self.model.get_depth() + 2))
        accuracies = []
        
        for depth in depths:
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
            temp_model.fit(X_test, y_test)  # 注意：这里用测试集是为了演示，实际应该用验证集
            pred = temp_model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            accuracies.append(acc)
        
        plt.plot(depths, accuracies, 'o-', linewidth=2, markersize=6)
        plt.axvline(x=self.model.get_depth(), color='red', linestyle='--', 
                   label=f'Current Depth: {self.model.get_depth()}')
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree - Depth vs Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 叶节点纯度分析
        plt.subplot(2, 4, 8)
        # 获取叶节点信息
        leaf_samples = []
        leaf_purities = []
        
        # 遍历所有叶节点
        for leaf_id in range(self.model.tree_.node_count):
            if self.model.tree_.children_left[leaf_id] == -1:  # 叶节点
                samples = self.model.tree_.n_node_samples[leaf_id]
                # 计算纯度（基尼系数）
                values = self.model.tree_.value[leaf_id][0]
                total = np.sum(values)
                if total > 0:
                    gini = 1 - np.sum((values / total) ** 2)
                    leaf_samples.append(samples)
                    leaf_purities.append(1 - gini)  # 转换为纯度
        
        if leaf_samples:
            plt.scatter(leaf_samples, leaf_purities, alpha=0.7, s=50)
            plt.xlabel('Samples in Leaf')
            plt.ylabel('Purity (1 - Gini)')
            plt.title('Decision Tree - Leaf Node Analysis')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'decision_tree_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存决策树规则
        self._save_decision_rules(save_path)
    
    def _save_decision_rules(self, save_path):
        """保存决策树规则"""
        # 导出决策树规则（文本格式）
        tree_rules = export_text(self.model, 
                               feature_names=self.feature_names if self.feature_names else None,
                               max_depth=10)  # 限制深度避免过长
        
        with open(os.path.join(save_path, 'decision_tree_rules.txt'), 'w', encoding='utf-8') as f:
            f.write("Decision Tree Rules:\n")
            f.write("=" * 50 + "\n")
            f.write(tree_rules)
        
        print(f"决策树规则已保存到 {os.path.join(save_path, 'decision_tree_rules.txt')}")
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型尚未训练"}
        
        info = {
            "model_type": "Decision Tree",
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "actual_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
            "n_nodes": self.model.tree_.node_count,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features
        }
        
        if self.feature_names is not None:
            feature_importance = self.model.feature_importances_
            top_features = [(self.feature_names[i], importance) 
                          for i, importance in enumerate(feature_importance)]
            top_features.sort(key=lambda x: x[1], reverse=True)
            info["top_5_features"] = top_features[:5]
        
        return info
    
    def analyze_overfitting(self, X_train, y_train, X_test, y_test, save_path='results'):
        """分析过拟合情况"""
        print("\n=== 过拟合分析 ===")
        
        plt.figure(figsize=(12, 8))
        
        # 测试不同的最大深度
        max_depths = range(1, 21)
        train_accuracies = []
        test_accuracies = []
        
        for depth in max_depths:
            # 训练不同深度的模型
            temp_model = DecisionTreeClassifier(
                max_depth=depth, 
                criterion=self.criterion,
                random_state=self.random_state
            )
            temp_model.fit(X_train, y_train)
            
            # 计算训练和测试准确率
            train_pred = temp_model.predict(X_train)
            test_pred = temp_model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
        
        # 绘制学习曲线
        plt.subplot(1, 2, 1)
        plt.plot(max_depths, train_accuracies, 'o-', label='Training Accuracy', linewidth=2)
        plt.plot(max_depths, test_accuracies, 'o-', label='Testing Accuracy', linewidth=2)
        plt.axvline(x=self.model.get_depth(), color='red', linestyle='--', 
                   label=f'Current Model Depth: {self.model.get_depth()}')
        plt.xlabel('Maximum Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree - Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算过拟合程度
        overfitting_scores = [train - test for train, test in zip(train_accuracies, test_accuracies)]
        
        plt.subplot(1, 2, 2)
        plt.plot(max_depths, overfitting_scores, 'o-', color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axvline(x=self.model.get_depth(), color='red', linestyle='--', 
                   label=f'Current Model Depth: {self.model.get_depth()}')
        plt.xlabel('Maximum Depth')
        plt.ylabel('Overfitting Score (Train Acc - Test Acc)')
        plt.title('Decision Tree - Overfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'decision_tree_overfitting.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 找到最佳深度
        best_depth_idx = np.argmax(test_accuracies)
        best_depth = max_depths[best_depth_idx]
        best_test_acc = test_accuracies[best_depth_idx]
        
        print(f"最佳深度: {best_depth}")
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        print(f"当前模型深度: {self.model.get_depth()}")
        print(f"当前模型过拟合分数: {overfitting_scores[self.model.get_depth()-1]:.4f}")
        
        return {
            'best_depth': best_depth,
            'best_test_accuracy': best_test_acc,
            'current_overfitting_score': overfitting_scores[min(self.model.get_depth()-1, len(overfitting_scores)-1)]
        }

if __name__ == "__main__":
    # 测试决策树分类器
    from data_loader import WineDataLoader
    
    # 加载数据
    loader = WineDataLoader()
    df = loader.load_data()
    data = loader.preprocess_data()
    info = loader.get_data_info()
    
    # 创建和训练决策树分类器
    dt = DecisionTreeClassifierCustom(criterion='gini', max_depth=10, random_state=42)
    dt.train(data['X_train'], data['y_train'], 
             feature_names=info['feature_names'],
             target_names=info['target_names'])
    
    # 预测和评估
    results = dt.predict(data['X_test'], data['y_test'])
    
    # 可视化结果
    dt.visualize_results(data['X_test'], data['y_test'], results)
    
    # 分析过拟合
    overfitting_analysis = dt.analyze_overfitting(
        data['X_train'], data['y_train'], 
        data['X_test'], data['y_test']
    )
    
    # 显示模型信息
    model_info = dt.get_model_info()
    print(f"\n模型信息: {model_info}")
    print(f"\n过拟合分析: {overfitting_analysis}")