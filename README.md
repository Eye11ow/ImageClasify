# ImageClasify
# CIFAR-10 图像分类神经网络

这个项目实现了一个三层神经网络模型用于CIFAR-10图像分类任务。该模型使用纯NumPy构建，无需深度学习框架，适合学习神经网络基础原理。

## 完整文件结构
```bash
.
├── README.md
├── main.py
├── data.py
├── model.py
├── train.py
├── test.py
├── hparam_search.py
└── vision.py
```

## 功能概述

该项目包含以下主要功能：

1. **数据加载与预处理**：从CIFAR-10数据集加载图像数据，并进行标准化和one-hot编码处理
2. **三层神经网络模型**：实现了带有一个隐藏层的神经网络，支持ReLU和Sigmoid激活函数
3. **模型训练**：使用SGD优化器训练模型，支持学习率衰减和L2正则化，应用了交叉熵损失
4. **模型评估**：在测试集上评估模型性能
5. **可视化功能**：绘制训练过程中的损失曲线和准确率变化，可视化学习到的权重

## 数据集准备

data.py中包含了数据加载和预处理的函数

## 使用方法

### 查找最优参数

通过以下命令来加载数据查找最优参数：

```bash
python main.py --mode search 
```

### 修改模型参数

可以在`main.py`中修改以下参数来调整模型：

- `hidden_size`：隐藏层的神经元数量
- `activation`：激活函数类型（'relu'或'tanh'）
- `epochs`：训练轮数
- `batch_size`：小批量大小
- `lr`：初始学习率
- `reg`：L2正则化系数
- `lr_decay`：学习率衰减因子

### 训练模型

通过指定参数训练模型：

```bash
python main.py --mode train --hidden_size 1024 --learning_rate 0.05 --reg 0.001
```

### 测试模型

利用训练得到的模型来测试数据，评估模型性能

```bash
python main.py --mode test --model_path best_model.npz
```

## 代码解释


### 神经网络模型

`ThreeLayerNN`类实现了三层神经网络模型，包括：

- `__init__`：初始化模型参数
- `forward`：前向传播
- `loss`：计算交叉熵损失和正则化损失
- `backward`：反向传播，计算梯度
- `pridict`：推理预测

### 训练与评估

- `train`函数实现了模型训练循环，包括SGD、学习率衰减和模型保存
- `test`函数在测试集上评估模型性能

### 可视化

- `plot_history`：绘制训练和验证损失曲线以及验证准确率曲线
- `visualize_weights`：可视化第一层权重