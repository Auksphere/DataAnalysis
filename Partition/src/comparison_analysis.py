"""
LDA和决策树分类方法对比分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, learning_curve
import time
import os

class ComparisonAnalysis:
    """LDA和决策树方法对比分析"""
    
    def __init__(self, lda_model, dt_model, feature_names=None, target_names=None):
        """
        初始化对比分析
        
        Parameters:
        - lda_model: 训练好的LDA模型
        - dt_model: 训练好的决策树模型
        - feature_names: 特征名称列表
        - target_names: 目标类别名称列表
        """
        self.lda_model = lda_model
        self.dt_model = dt_model
        self.feature_names = feature_names
        self.target_names = target_names
        self.comparison_results = {}
        
    def compare_performance(self, X_test, y_test, X_train=None, y_train=None):
        """比较模型性能"""
        print("=== 模型性能对比分析 ===")
        
        # LDA预测
        start_time = time.time()
        lda_pred = self.lda_model.predict(X_test)
        lda_pred_proba = self.lda_model.predict_proba(X_test)
        lda_time = time.time() - start_time
        
        # 决策树预测
        start_time = time.time()
        dt_pred = self.dt_model.predict(X_test)
        dt_pred_proba = self.dt_model.predict_proba(X_test)
        dt_time = time.time() - start_time
        
        # 计算性能指标
        metrics = {}
        
        # LDA指标
        lda_accuracy = accuracy_score(y_test, lda_pred)
        lda_precision, lda_recall, lda_f1, _ = precision_recall_fscore_support(y_test, lda_pred, average='weighted')
        
        # 决策树指标
        dt_accuracy = accuracy_score(y_test, dt_pred)
        dt_precision, dt_recall, dt_f1, _ = precision_recall_fscore_support(y_test, dt_pred, average='weighted')
        
        # 多分类AUC计算
        n_classes = len(np.unique(y_test))
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            lda_auc = roc_auc_score(y_test_bin, lda_pred_proba, multi_class='ovr', average='weighted')
            dt_auc = roc_auc_score(y_test_bin, dt_pred_proba, multi_class='ovr', average='weighted')
        else:
            lda_auc = roc_auc_score(y_test, lda_pred_proba[:, 1])
            dt_auc = roc_auc_score(y_test, dt_pred_proba[:, 1])
        
        metrics = {
            'LDA': {
                'accuracy': lda_accuracy,
                'precision': lda_precision,
                'recall': lda_recall,
                'f1_score': lda_f1,
                'auc': lda_auc,
                'prediction_time': lda_time,
                'predictions': lda_pred,
                'probabilities': lda_pred_proba
            },
            'Decision_Tree': {
                'accuracy': dt_accuracy,
                'precision': dt_precision,
                'recall': dt_recall,
                'f1_score': dt_f1,
                'auc': dt_auc,
                'prediction_time': dt_time,
                'predictions': dt_pred,
                'probabilities': dt_pred_proba
            }
        }
        
        # 打印对比结果
        print(f"\n性能对比结果:")
        print(f"{'指标':<15} {'LDA':<10} {'决策树':<10} {'差异':<10}")
        print("-" * 50)
        print(f"{'准确率':<15} {lda_accuracy:<10.4f} {dt_accuracy:<10.4f} {abs(lda_accuracy-dt_accuracy):<10.4f}")
        print(f"{'精确率':<15} {lda_precision:<10.4f} {dt_precision:<10.4f} {abs(lda_precision-dt_precision):<10.4f}")
        print(f"{'召回率':<15} {lda_recall:<10.4f} {dt_recall:<10.4f} {abs(lda_recall-dt_recall):<10.4f}")
        print(f"{'F1分数':<15} {lda_f1:<10.4f} {dt_f1:<10.4f} {abs(lda_f1-dt_f1):<10.4f}")
        print(f"{'AUC':<15} {lda_auc:<10.4f} {dt_auc:<10.4f} {abs(lda_auc-dt_auc):<10.4f}")
        print(f"{'预测时间(s)':<15} {lda_time:<10.6f} {dt_time:<10.6f} {abs(lda_time-dt_time):<10.6f}")
        
        self.comparison_results['performance'] = metrics
        
        # 交叉验证对比（如果提供训练数据）
        if X_train is not None and y_train is not None:
            self._cross_validation_comparison(X_train, y_train)
        
        return metrics
    
    def _cross_validation_comparison(self, X_train, y_train, cv=5):
        """交叉验证对比"""
        print(f"\n=== {cv}折交叉验证对比 ===")
        
        # LDA交叉验证
        lda_cv_scores = cross_val_score(self.lda_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # 决策树交叉验证
        dt_cv_scores = cross_val_score(self.dt_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"LDA交叉验证准确率: {lda_cv_scores.mean():.4f} (+/- {lda_cv_scores.std() * 2:.4f})")
        print(f"决策树交叉验证准确率: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")
        
        self.comparison_results['cross_validation'] = {
            'lda_scores': lda_cv_scores,
            'dt_scores': dt_cv_scores,
            'lda_mean': lda_cv_scores.mean(),
            'dt_mean': dt_cv_scores.mean(),
            'lda_std': lda_cv_scores.std(),
            'dt_std': dt_cv_scores.std()
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性对比"""
        print("\n=== 特征重要性对比分析 ===")
        
        if self.feature_names is None:
            print("未提供特征名称，跳过特征重要性分析")
            return
        
        feature_analysis = {}
        
        # LDA特征重要性（基于判别函数系数）
        if hasattr(self.lda_model, 'coef_'):
            lda_importance = np.abs(self.lda_model.coef_[0])  # 取第一个判别函数
            lda_importance = lda_importance / np.sum(lda_importance)  # 归一化
            feature_analysis['lda_importance'] = lda_importance
        
        # 决策树特征重要性
        dt_importance = self.dt_model.feature_importances_
        feature_analysis['dt_importance'] = dt_importance
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'LDA_Importance': lda_importance if 'lda_importance' in feature_analysis else np.zeros(len(self.feature_names)),
            'DT_Importance': dt_importance
        })
        
        # 计算相关性
        if 'lda_importance' in feature_analysis:
            correlation = np.corrcoef(lda_importance, dt_importance)[0, 1]
            print(f"LDA和决策树特征重要性相关性: {correlation:.4f}")
            feature_analysis['importance_correlation'] = correlation
        
        self.comparison_results['feature_importance'] = feature_analysis
        
        return importance_df
    
    def visualize_comparison(self, X_test, y_test, save_path='results'):
        """可视化对比结果"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 性能指标对比雷达图
        plt.subplot(2, 4, 1)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        lda_values = [
            self.comparison_results['performance']['LDA']['accuracy'],
            self.comparison_results['performance']['LDA']['precision'],
            self.comparison_results['performance']['LDA']['recall'],
            self.comparison_results['performance']['LDA']['f1_score'],
            self.comparison_results['performance']['LDA']['auc']
        ]
        dt_values = [
            self.comparison_results['performance']['Decision_Tree']['accuracy'],
            self.comparison_results['performance']['Decision_Tree']['precision'],
            self.comparison_results['performance']['Decision_Tree']['recall'],
            self.comparison_results['performance']['Decision_Tree']['f1_score'],
            self.comparison_results['performance']['Decision_Tree']['auc']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, lda_values, width, label='LDA', alpha=0.8)
        bars2 = plt.bar(x + width/2, dt_values, width, label='Decision Tree', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 混淆矩阵对比
        plt.subplot(2, 4, 2)
        lda_cm = confusion_matrix(y_test, self.comparison_results['performance']['LDA']['predictions'])
        lda_cm_norm = lda_cm.astype('float') / lda_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(lda_cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.title('LDA - Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.subplot(2, 4, 3)
        dt_cm = confusion_matrix(y_test, self.comparison_results['performance']['Decision_Tree']['predictions'])
        dt_cm_norm = dt_cm.astype('float') / dt_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(dt_cm_norm, annot=True, fmt='.2f', cmap='Greens', cbar=False)
        plt.title('Decision Tree - Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 3. 特征重要性对比
        plt.subplot(2, 4, 4)
        if 'feature_importance' in self.comparison_results:
            importance_data = self.comparison_results['feature_importance']
            if 'lda_importance' in importance_data:
                # 选择前10个最重要的特征（按决策树重要性排序）
                top_indices = np.argsort(importance_data['dt_importance'])[-10:]
                
                x = np.arange(len(top_indices))
                width = 0.35
                
                lda_top = importance_data['lda_importance'][top_indices]
                dt_top = importance_data['dt_importance'][top_indices]
                feature_names_top = np.array(self.feature_names)[top_indices]
                
                plt.barh(x - width/2, lda_top, width, label='LDA', alpha=0.8)
                plt.barh(x + width/2, dt_top, width, label='Decision Tree', alpha=0.8)
                
                plt.yticks(x, [name[:15] + '...' if len(name) > 15 else name for name in feature_names_top])
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Feature Importance Comparison')
                plt.legend()
        
        # 4. 预测概率分布对比
        plt.subplot(2, 4, 5)
        lda_proba = self.comparison_results['performance']['LDA']['probabilities']
        dt_proba = self.comparison_results['performance']['Decision_Tree']['probabilities']
        
        # 计算最大预测概率的分布
        lda_max_proba = np.max(lda_proba, axis=1)
        dt_max_proba = np.max(dt_proba, axis=1)
        
        plt.hist(lda_max_proba, alpha=0.7, label='LDA', bins=30, density=True)
        plt.hist(dt_max_proba, alpha=0.7, label='Decision Tree', bins=30, density=True)
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        
        # 5. 交叉验证结果对比（如果有）
        plt.subplot(2, 4, 6)
        if 'cross_validation' in self.comparison_results:
            cv_data = self.comparison_results['cross_validation']
            
            box_data = [cv_data['lda_scores'], cv_data['dt_scores']]
            box_labels = ['LDA', 'Decision Tree']
            
            bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            
            plt.ylabel('Cross-Validation Accuracy')
            plt.title('Cross-Validation Results Comparison')
            plt.grid(True, alpha=0.3)
        
        # 6. 预测时间对比
        plt.subplot(2, 4, 7)
        time_data = [
            self.comparison_results['performance']['LDA']['prediction_time'],
            self.comparison_results['performance']['Decision_Tree']['prediction_time']
        ]
        models = ['LDA', 'Decision Tree']
        
        bars = plt.bar(models, time_data, color=['lightblue', 'lightgreen'], alpha=0.8)
        plt.ylabel('Prediction Time (seconds)')
        plt.title('Prediction Time Comparison')
        
        # 添加数值标签
        for bar, time_val in zip(bars, time_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_data)*0.01,
                    f'{time_val:.6f}s', ha='center', va='bottom')
        
        # 7. 错误分析
        plt.subplot(2, 4, 8)
        # 分析两个模型的错误分布
        lda_errors = y_test != self.comparison_results['performance']['LDA']['predictions']
        dt_errors = y_test != self.comparison_results['performance']['Decision_Tree']['predictions']
        
        # 计算不同类型的错误
        both_correct = (~lda_errors) & (~dt_errors)
        both_wrong = lda_errors & dt_errors
        lda_wrong_only = lda_errors & (~dt_errors)
        dt_wrong_only = (~lda_errors) & dt_errors
        
        error_counts = [
            np.sum(both_correct),
            np.sum(both_wrong),
            np.sum(lda_wrong_only),
            np.sum(dt_wrong_only)
        ]
        error_labels = ['Both Correct', 'Both Wrong', 'LDA Wrong Only', 'DT Wrong Only']
        colors = ['green', 'red', 'blue', 'orange']
        
        plt.pie(error_counts, labels=error_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Error Analysis Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存详细对比报告
        self._save_comparison_report(save_path)
    
    def _save_comparison_report(self, save_path):
        """保存详细对比报告"""
        report_path = os.path.join(save_path, 'comparison_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LDA vs Decision Tree 分类方法对比报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 性能对比
            f.write("1. 性能指标对比\n")
            f.write("-" * 20 + "\n")
            perf = self.comparison_results['performance']
            
            f.write(f"准确率:\n")
            f.write(f"  LDA: {perf['LDA']['accuracy']:.4f}\n")
            f.write(f"  决策树: {perf['Decision_Tree']['accuracy']:.4f}\n")
            f.write(f"  差异: {abs(perf['LDA']['accuracy'] - perf['Decision_Tree']['accuracy']):.4f}\n\n")
            
            f.write(f"精确率 (加权平均):\n")
            f.write(f"  LDA: {perf['LDA']['precision']:.4f}\n")
            f.write(f"  决策树: {perf['Decision_Tree']['precision']:.4f}\n\n")
            
            f.write(f"召回率 (加权平均):\n")
            f.write(f"  LDA: {perf['LDA']['recall']:.4f}\n")
            f.write(f"  决策树: {perf['Decision_Tree']['recall']:.4f}\n\n")
            
            f.write(f"F1分数 (加权平均):\n")
            f.write(f"  LDA: {perf['LDA']['f1_score']:.4f}\n")
            f.write(f"  决策树: {perf['Decision_Tree']['f1_score']:.4f}\n\n")
            
            f.write(f"AUC:\n")
            f.write(f"  LDA: {perf['LDA']['auc']:.4f}\n")
            f.write(f"  决策树: {perf['Decision_Tree']['auc']:.4f}\n\n")
            
            f.write(f"预测时间:\n")
            f.write(f"  LDA: {perf['LDA']['prediction_time']:.6f} 秒\n")
            f.write(f"  决策树: {perf['Decision_Tree']['prediction_time']:.6f} 秒\n\n")
            
            # 交叉验证结果
            if 'cross_validation' in self.comparison_results:
                f.write("2. 交叉验证结果\n")
                f.write("-" * 20 + "\n")
                cv = self.comparison_results['cross_validation']
                f.write(f"LDA: {cv['lda_mean']:.4f} (+/- {cv['lda_std'] * 2:.4f})\n")
                f.write(f"决策树: {cv['dt_mean']:.4f} (+/- {cv['dt_std'] * 2:.4f})\n\n")
            
            # 特征重要性分析
            if 'feature_importance' in self.comparison_results:
                f.write("3. 特征重要性分析\n")
                f.write("-" * 20 + "\n")
                if 'importance_correlation' in self.comparison_results['feature_importance']:
                    corr = self.comparison_results['feature_importance']['importance_correlation']
                    f.write(f"特征重要性相关性: {corr:.4f}\n\n")
        
        print(f"详细对比报告已保存到: {report_path}")
    
    def generate_analysis_summary(self):
        """生成分析总结"""
        print("\n=== 分析总结 ===")
        
        if not self.comparison_results:
            print("请先运行性能对比分析")
            return
        
        perf = self.comparison_results['performance']
        
        # 确定最佳模型
        lda_acc = perf['LDA']['accuracy']
        dt_acc = perf['Decision_Tree']['accuracy']
        
        if lda_acc > dt_acc:
            best_model = "LDA"
            best_acc = lda_acc
            diff = lda_acc - dt_acc
        else:
            best_model = "决策树"
            best_acc = dt_acc
            diff = dt_acc - lda_acc
        
        print(f"最佳模型: {best_model}")
        print(f"最佳准确率: {best_acc:.4f}")
        print(f"准确率差异: {diff:.4f}")
        
        # 速度对比
        lda_time = perf['LDA']['prediction_time']
        dt_time = perf['Decision_Tree']['prediction_time']
        
        if lda_time < dt_time:
            faster_model = "LDA"
            time_diff = dt_time - lda_time
        else:
            faster_model = "决策树"
            time_diff = lda_time - dt_time
        
        print(f"\n速度更快的模型: {faster_model}")
        print(f"时间差异: {time_diff:.6f} 秒")
        
        # 总体建议
        print(f"\n总体建议:")
        if diff > 0.05:  # 准确率差异较大
            print(f"- 推荐使用 {best_model}，在准确率上有明显优势")
        else:
            print(f"- 两种方法性能相近，可根据具体需求选择")
        
        if time_diff > 0.001:  # 时间差异较大
            print(f"- 如果对预测速度有要求，推荐使用 {faster_model}")
        
        return {
            'best_accuracy_model': best_model,
            'best_accuracy': best_acc,
            'accuracy_difference': diff,
            'faster_model': faster_model,
            'time_difference': time_diff
        }

if __name__ == "__main__":
    # 这个文件主要作为模块使用，不直接运行
    print("ComparisonAnalysis 模块已加载")