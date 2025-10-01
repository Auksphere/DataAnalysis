"""
线性回归分析主程序
运行完整的线性回归分析流程
"""
import os
import sys
import time
from datetime import datetime

def print_header():
    """打印程序头部信息"""
    print("=" * 80)
    print("                    线性回归分析系统")
    print("                  Linear Regression Analysis")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("数据集: 加州房价数据集")
    print("分析内容: 一元/多元线性回归 + 三种采样方法比较")
    print("=" * 80)

def run_complete_analysis():
    """运行完整的分析流程"""
    
    print_header()
    
    try:
        # 1. 数据预处理
        print("步骤 1/4: 数据预处理...")
        print("-" * 50)
        from src.data_preprocessing import load_and_prepare_data, explore_data, preprocess_data, save_processed_data
        
        # 加载和预处理数据
        data, feature_names = load_and_prepare_data()
        explore_data(data, feature_names)
        X_processed, y_processed, scaler = preprocess_data(data, feature_names)
        save_processed_data(X_processed, y_processed, feature_names)
        print("✓ 数据预处理完成")
        
        # 2. 采样方法测试
        print("步骤 2/4: 采样方法测试...")
        print("-" * 50)
        from src.sampling_methods import SamplingMethods
        
        sampler = SamplingMethods(X_processed, y_processed)
        
        # 测试三种采样方法
        print("测试留出法...")
        X_train, X_val, X_test, y_train, y_val, y_test = sampler.holdout_split()
        
        print("测试K折交叉验证...")
        kfold_splits = list(sampler.k_fold_split(k=5))
        
        print("测试自助法...")
        bootstrap_samples = sampler.bootstrap_sample(n_bootstraps=50)
        print("✓ 采样方法测试完成")
        
        # 3. 模型分析
        print("步骤 3/4: 线性回归模型分析...")
        print("-" * 50)
        
        # 一元线性回归
        print("运行一元线性回归分析...")
        try:
            from src.univariate_regression import run_univariate_analysis
            univariate_results = run_univariate_analysis(X_processed, y_processed, feature_names)
        except Exception as e:
            print(f"一元回归导入错误: {e}")
            raise e
        
        # 多元线性回归
        print("运行多元线性回归分析...")
        from src.multivariate_regression import run_multivariate_analysis
        multivariate_results = run_multivariate_analysis(X_processed, y_processed, feature_names)
        print("✓ 模型分析完成")
        
        # 4. 综合分析和可视化
        print("步骤 4/4: 综合分析和报告生成...")
        print("-" * 50)
        from src.comprehensive_analysis import create_comprehensive_visualization, generate_analysis_summary, print_comprehensive_report
        
        # 生成可视化
        print("生成综合可视化图表...")
        visualization_path = create_comprehensive_visualization(univariate_results, multivariate_results)
        
        # 生成分析摘要
        print("生成分析摘要...")
        summary = generate_analysis_summary(univariate_results, multivariate_results)
        
        # 打印综合报告
        print("生成综合报告...")
        print_comprehensive_report(summary)
        print("✓ 综合分析完成")
        
        # 总结
        print("" + "=" * 80)
        print("                        分析完成!")
        print("=" * 80)
        print("生成的文件:")
        print(f"1. 数据探索图表: results/data_exploration.png")
        print(f"2. 综合分析图表: results/comprehensive_analysis.png")
        print(f"3. 处理后数据: data/processed_data.csv")
        print(f"4. 分析报告: LinearRegression_Analysis_Report.md")
        
        print("主要结论:")
        print(f"- 最佳一元特征: {univariate_results['feature']} (相关性: {univariate_results['correlation']:.3f})")
        print(f"- 一元回归测试集R²: {univariate_results['holdout_scores']['test_r2']:.4f}")
        print(f"- 多元回归测试集R²: {multivariate_results['holdout_scores']['test_r2']:.4f}")
        print(f"- 性能提升: +{(multivariate_results['holdout_scores']['test_r2'] - univariate_results['holdout_scores']['test_r2']):.4f}")
        print(f"- 最重要特征: {multivariate_results['feature_importance'].iloc[0]['feature']}")
        
        print("采样方法总结:")
        print("- K折交叉验证: 最稳定，推荐用于模型评估")
        print("- 留出法: 快速简单，适合初步分析")
        print("- 自助法: 适合小数据集和不确定性估计")
        
        return {
            'univariate_results': univariate_results,
            'multivariate_results': multivariate_results,
            'summary': summary,
            'visualization_path': visualization_path
        }
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        print("请检查数据和代码，然后重新运行。")
        return None

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下必需包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✓ 环境检查通过")
    return True

if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行完整分析
    results = run_complete_analysis()
    
    # 计算运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    
    if results:
        print(f"总运行时间: {execution_time:.2f} 秒")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("线性回归分析全部完成！")
    else:
        print(f"❌ 分析失败，运行时间: {execution_time:.2f} 秒")