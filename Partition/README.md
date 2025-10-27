# 分类方法比较分析项目

## 项目概述
本项目实现并比较了两种经典的分类方法：
1. **线性判别分析 (Linear Discriminant Analysis, LDA)**
2. **决策树 (Decision Tree)**

## 数据集
使用**Wine质量数据集**，这是一个包含红酒理化特性的真实数据集，目标是根据理化指标预测红酒质量等级。

## 项目结构
```
Partition/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── main.py                      # 主程序入口
├── src/
│   ├── data_loader.py          # 数据加载和预处理
│   ├── lda_classifier.py       # LDA分类器实现
│   ├── decision_tree_classifier.py  # 决策树分类器实现
│   └── comparison_analysis.py   # 方法对比分析
├── data/                        # 数据文件夹
├── results/                     # 结果输出文件夹
└── Classification_Analysis_Report.md  # 分析报告

```

## 运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python main.py`
3. 查看结果：检查 `results/` 文件夹中的输出文件和图表

## 分析内容
- 数据预处理和探索性数据分析
- LDA分类模型训练和评估
- 决策树分类模型训练和评估
- 两种方法的性能对比
- 优势和局限性分析