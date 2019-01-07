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
import csv

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

batch_size = 50
img_size = 224

# pytorch内置预训练的模型，需要图片大小至少为224X224
transform = T.Compose([
    T.Resize(img_size),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(img_size),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
])

# 读取内置的resnet101预训练模型及其参数
resnet101 = models.resnet101(pretrained=True)
resnet101.fc = nn.Linear(2048, 5)  # 替换最后一层为2分类，原本为ImageNet的1000类分类任务
net = resnet101  # 赋值给net，统一化命名

if torch.cuda.is_available():
    print("Using GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

train_dataset = ImageFolder('./data/train/train', transform=transform)
test_dataset = ImageFolder('./data/test/', transform=transform)


trainloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8)
testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1)

# CSV, result submit
headers = ['Id', 'Expected']
rows = []


def test():
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    result_index = 0
    net.eval()

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            index = np.asarray(predicted.cpu().clone())
            for i in range(len(predicted)):
                b = int(index[i])
                label_gt = classes[b]
                rows.append({'Id':result_index, 'Expected':label_gt})
                result_index += 1

        with open('result.csv', 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(rows)


criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

print("Starting to train")
torch.set_num_threads(4)
for epoch in range(50):
    net.train()

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

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
        print_gap = 10
        if i % print_gap == (print_gap-1): # 每print_gap个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / print_gap))
            running_loss = 0.0

print('Finished Training')

print("Starting to test")
test()

