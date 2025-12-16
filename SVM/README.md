# SVM与AdaBoost分析

本项目在西瓜数据集上实现并比较了不同核函数的支持向量机(SVM)和AdaBoost集成学习方法。

## 数据集

数据集(`dataset.csv`)包含西瓜样本，具有以下特征：
- **密度(Density)**：西瓜的密度
- **含糖量(Sugar content)**：含糖量测量值
- **标签(Label)**：二分类（1 = 好瓜，0 = 坏瓜）

总样本数：17个（8个好瓜，9个坏瓜）

## 项目结构

```
SVM/
├── dataset.csv                     # 输入数据集
├── svm_analysis.py                 # SVM实现与分析
├── adaboost_analysis.py           # AdaBoost实现与分析
├── main.py                        # 主程序入口
├── requirements.txt               # Python依赖包
├── README.md                      # 本文件
└── results/                       # 输出可视化结果
    ├── svm_comparison.png
    ├── adaboost_comparison.png
    └── adaboost_individual_learners.png
```

## 实现任务

### 任务1：不同核函数的SVM

使用LIBSVM（通过scikit-learn）实现两种核函数类型的SVM：
1. **线性核(Linear Kernel)**：创建线性决策边界
2. **高斯核(Gaussian/RBF Kernel)**：创建非线性决策边界

**关键分析：**
- 在西瓜数据集上训练两种SVM
- 识别并比较支持向量
- 可视化决策边界和间隔
- 分析支持向量选择的差异

**主要区别：**
- 线性核通常对线性可分数据使用较少的支持向量
- RBF核更灵活，可以建模复杂模式
- RBF核通常需要更多支持向量
- 线性核对简单模式提供更好的泛化能力

### 任务2：AdaBoost集成学习

从零实现AdaBoost算法，使用不剪枝决策树作为基学习器。

**关键特性：**
- 自定义AdaBoost实现（未使用sklearn的AdaBoost）
- 使用不剪枝决策树作为弱学习器
- 比较不同集成规模：1、3、5和11个基学习器
- 可视化决策边界随集成规模的演变

**关键分析：**
- 展示各个基学习器的决策边界
- 演示集成相对于单个分类器的改进
- 说明边界如何随更多学习器变得更精细
- 跟踪准确率随集成增长的改进

## 安装

1. 确保已激活conda环境：
```bash
conda activate Data
```

2. 安装所需包：
```bash
pip install -r requirements.txt
```

## 使用方法

运行完整分析：
```bash
python main.py
```

这将：
1. 运行线性核和高斯核的SVM分析
2. 运行不同集成规模的AdaBoost分析
3. 在`results/`目录中生成可视化结果
4. 在控制台打印详细对比报告

### 单独运行分析

**仅运行SVM分析：**
```bash
python svm_analysis.py
```

**仅运行AdaBoost分析：**
```bash
python adaboost_analysis.py
```

## 输出结果

分析生成三个主要可视化图：

1. **svm_comparison.png**
   - 线性核和RBF核SVM的并排比较
   - 显示决策边界、间隔和支持向量
   - 突出显示支持向量选择的差异

2. **adaboost_comparison.png**
   - 比较1、3、5和11个基学习器的AdaBoost性能
   - 显示决策边界的演变
   - 显示每个集成规模的准确率

3. **adaboost_individual_learners.png**
   - 可视化前4个独立的基学习器
   - 显示每个学习器的决策边界和权重(α)
   - 说明弱学习器如何组合形成强分类器

## 技术细节

### SVM实现
- 使用`sklearn.svm.SVC`（LIBSVM实现）
- 使用`StandardScaler`标准化特征
- 默认参数：C=1.0，RBF核的gamma='scale'

### AdaBoost实现
- 遵循AdaBoost.M1算法的自定义实现
- 基学习器：不剪枝决策树（`max_depth=None`）
- 权重更新公式：w_t+1 = w_t * exp(-α_t * y * h_t(x))
- Alpha计算：α_t = 0.5 * log((1 - ε_t) / ε_t)

## 依赖包

- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

## 注意事项

- 所有图表使用英文标签，遵循标准可视化规范
- 数据集较小（17个样本），结果应相应解释
- 特征已标准化以提高模型性能
- 设置随机种子以保证可重复性（random_state=42）

## 作者

DataAnalysis项目 - SVM与AdaBoost模块
