import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

data_dir = 'G:/BaiduNetdiskDownload/dataset/cifar-10'

# 整理数据集
def read_csv_labels(fname):
    """读取 'fname' 来给标签字典返回一个文件名。"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:] # 一行一行读进来，每一行为列表中一个元素
    tokens = [l.rstrip().split(',') for l in lines] # 遍历列表每一个元素，切分
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
# 将验证集从原始的训练集中拆分出来
# train文件夹下有所有train的图片，test文件夹下有所有test图片
# 把train文件夹下所有类的图片创建一个类名文件夹，然后搬到对应文件夹下
def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

# 在预测期间整理测试集，以方便读取
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))  # unknown为 test文件夹里面的一个文件夹

# 调用前面定义的函数，前面只是定义函数，这里是调用
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

valid_ratio = 0.1  # train 数据里面百分之九十用来训练，剩下百分之十用来验证
reorg_cifar10_data(data_dir, valid_ratio)
