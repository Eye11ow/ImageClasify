import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical

def load_and_preprocess_data():
    # 加载 CIFAR-10 数据集
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # 归一化像素值到 [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # 展平图像为一维向量
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    # 将标签转换为 one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # 分割一部分训练集作为验证集
    split_index = int(0.8 * len(X_train))
    X_val = X_train[split_index:]
    y_val = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    
    return X_train, y_train, X_val, y_val, X_test, y_test