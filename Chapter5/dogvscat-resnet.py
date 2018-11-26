"""
GPU and CPU version that use pretrained ResNet to
deal with the dog-vs-cat problem
Using GPU or CPU depends on whether cuda is available
"""
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import torchvision as tv
from torchvision.transforms import ToPILImage
from torchvision import models

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# pytorch内置预训练的模型，需要图片大小至少为224X224
transform = T.Compose([
    T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
])

# 读取内置的resnet34预训练模型及其参数
resnet34 = models.resnet34(pretrained=True)
resnet34.fc = nn.Linear(512, 2)  # 替换最后一层为2分类，原本为ImageNet的1000类分类任务
net = resnet34  # 赋值给net，统一化命名

if torch.cuda.is_available():
    print("Using GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 打印restnet34各个层的信息
# for name,parameters in net.named_parameters():
#     print(name,':',parameters.size())

train_dataset = ImageFolder('/home/lzx/datasets/dogcat/sub-train/', transform=transform)
test_dataset = ImageFolder('/home/lzx/datasets/dogcat/sub-test/', transform=transform)
# dataset = DogCat('/home/lzx/datasets/dogcat/sub-train/', transforms=transform)
# train_dataset = ImageFolder('/Users/lizhixuan/PycharmProjects/pytorch_learning/Chapter5/sub-train/', transform=transform)
# test_dataset = ImageFolder('/Users/lizhixuan/PycharmProjects/pytorch_learning/Chapter5/sub-test/', transform=transform)


trainloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=128,
                    shuffle=True,
                    num_workers=4)
testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=2)


def test():
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy in the test dataset: %d %%' % (100 * correct / total))

classes = ('cat', 'dog')

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

print("Starting to train")
torch.set_num_threads(4)
for epoch in range(1000):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        inputs, labels = data
        # inputs = Variable(inputs)
        # labels = Variable(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        print_gap = 100
        if i % print_gap == (print_gap-1): # 每print_gap个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / print_gap))
            running_loss = 0.0
    test()
print('Finished Training')
