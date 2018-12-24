# Chapter 5
+ 用自定义的LeNet来实现猫狗二分类器
+ 用预训练好的ResNet34来实现猫狗二分类器


## 1. 用自定义的LeNet来实现猫狗二分类器
+ 数据集：完整的猫狗二分类数据集。猫狗二分类问题是kaggle上的一个经典问题，详见：[https://github.com/chenyuntc/pytorch-book/tree/master/chapter6-实战指南](https://github.com/chenyuntc/pytorch-book/tree/master/chapter6-实战指南)
+ 猫和狗都分别有12500张训练数据，这里只用了训练集。其中10000张x2类用于训练，2500张x2类用于测试
+ 训练轮数：1000个epoch
+ 学习率：0.001
+ 最终的准确率：87%


## 2. 用预训练好的ResNet34来实现猫狗二分类器
+ 除了把模型更换为与训练好的ResNet34，别的都基本一样
+ 猫和狗都分别有12500张训练数据，这里只用了训练集。其中10000张x2类用于训练，2500张x2类用于测试
+ 学习率：0.0001
+ 训练轮数：1000个epoch
+ 最终的准确率：