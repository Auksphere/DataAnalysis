# 卷积神经网络图像分类项目

## 项目概述
本项目实现了基于卷积神经网络(CNN)的图像分类系统，使用CIFAR-10数据集进行训练和测试。

## 文件结构
```
NeuroNetwork/
├── README.md                     # 项目说明
├── requirements.txt               # 依赖包列表
├── main.py                       # 主程序入口
├── config.py                     # 配置文件
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载模块
│   ├── cnn_models.py             # CNN模型定义
│   ├── train.py                  # 训练模块
│   ├── evaluate.py               # 评估模块
│   └── utils.py                  # 工具函数
├── models/                       # 模型保存目录
├── results/                      # 结果保存目录
│   ├── plots/                    # 图表保存
│   └── reports/                  # 报告保存
└── CNN_分析报告.md               # 详细分析报告
```

## 数据集
- CIFAR-10: 包含10个类别的60000张32x32彩色图像
- 类别: 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

## 运行方式
1. 安装依赖: `pip install -r requirements.txt`
2. 运行主程序: `python main.py`

## 主要特性
- 多种CNN架构对比
- 不同卷积核尺寸实验
- 详细的性能分析
- 可视化结果展示