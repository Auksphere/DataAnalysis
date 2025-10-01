# 线性回归分析项目(大数据分析方法第一次作业)

本项目基于加州房价数据集，采用一元和多元线性回归对数据进行分析，使用三种数据采样方法（留出法、K折交叉验证法、自助法）进行模型评估和比较。

## 项目结构

```
LinearRegression/
├── data/           # 数据文件
├── src/            # 源代码
├── models/         # 模型文件
├── results/        # 结果和图表
├── requirements.txt # 依赖包
└── README.md       # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

0. ```bash
    cd LinearRegression/
    ```
1. 运行数据预处理：`python src/data_preprocessing.py`
2. 执行采样方法比较：`python src/sampling_methods.py`
3. 运行线性回归分析：`python src/conprehensive_analysis.py`
4. 一键运行：`python main.py`

## 分析内容

- 一元线性回归分析
- 多元线性回归分析
- 三种采样方法比较（留出法、K折交叉验证、自助法）
- 模型性能评估和可视化

## 结果查看
程序执行结果储存在./results路径下，分析报告见LinearRegression_Analysis_Report.md。