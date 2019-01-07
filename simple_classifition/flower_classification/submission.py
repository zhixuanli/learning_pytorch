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

from dataloader import ImageFolderWithPaths

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

# 读取内置的densenet201预训练模型及其参数
densenet201 = models.densenet201(pretrained=True)
num_ftrs = densenet201.classifier.in_features
densenet201.classifier = nn.Linear(num_ftrs, 5)  # 替换最后一层为5分类，原本为ImageNet的1000类分类任务
net = densenet201  # 赋值给net，统一化命名

if torch.cuda.is_available():
    print("Using GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# train_dataset = ImageFolder('./data/train/train', transform=transform)
# test_dataset = ImageFolder('./data/test/', transform=transform)
train_dataset = ImageFolderWithPaths('./data/train/train', transform=transform)
test_dataset = ImageFolderWithPaths('./data/test/', transform=transform)


trainloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8)
testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4)

# CSV, result submit
headers = ['Id', 'Expected']
rows = []

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
epoch_s = 0

if os.path.exists('./log/state.pkl'):
    checkpoint = torch.load('./log/state.pkl')
    # print(checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_s = checkpoint['epoch']
    print("checkpoint loaded")


def test():
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    result_index = 0
    net.eval()

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with torch.no_grad():
        # for data in testloader:
        for i, data in enumerate(testloader, 0):
            images, labels, paths = data
            # result_index = i*4

            images = images.to(device)

            root, _ = os.path.splitext(paths[0])
            img_index = root.split("/")
            result_index = int(img_index[4])

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            index = np.asarray(predicted.cpu().clone())
            for i in range(len(predicted)):
                print(result_index)
                b = int(index[i])
                label_gt = classes[b]
                rows.append({'Id':result_index, 'Expected':label_gt})
                # result_index += 1

        with open('result.csv', 'w') as f:
            f_csv = csv.DictWriter(f, headers)
            f_csv.writeheader()
            f_csv.writerows(rows)


def get_learning_rate(epoch):
    if epoch < 50:
        lr = 1e-4
    else:
        lr = 1e-5
    return lr


def train():
    print("Starting to train")
    torch.set_num_threads(4)
    for epoch in range(epoch_s, 100):
        net.train()

        lr = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("learning rate = %f" % lr)

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

    torch.save({
                'epoch': epoch+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "./log/state.pkl")

    print('Finished Training')

train()

print("Starting to test")
test()

