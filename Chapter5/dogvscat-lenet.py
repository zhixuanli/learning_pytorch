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
show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# tranform to resize
# 原本的图像（猫狗数据集）是不整齐的，
# 不同图像的大小并不相同
transform = T.Compose([
    T.Resize(32), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(32), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
])


# 定义LeNet
# 先定义网络结构，但是并没有规定它们如何组织在一起，只是给出了网络都有哪些模块
# 再定义前向传播的过程，相当于把网络结构搭建起来
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 根据类来声明一个实例
net = Net()

# 如果GPU可用的话，设定device（设备）为GPU
# 否则设定为CPU，相当于自适应
if torch.cuda.is_available():
    print("Using GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)  # 把网络移动到设备上进行运算；本质是把参数移到了设备上

# 定义训练集和测试集，这里还没有定义验证集（还没学会）
# ImageFolder方法使用：在sub-train下分别有dog和cat两个文件夹放着对应的图片，
# pytorch会自动识别dog和cat为这些图片的label信息
train_dataset = ImageFolder('/home/lzx/datasets/dogcat/sub-train/', transform=transform)
test_dataset = ImageFolder('/home/lzx/datasets/dogcat/sub-test/', transform=transform)
# dataset = DogCat('/home/lzx/datasets/dogcat/sub-train/', transforms=transform)
# train_dataset = ImageFolder('/Users/lizhixuan/PycharmProjects/pytorch_learning/Chapter5/sub-train/', transform=transform)
# test_dataset = ImageFolder('/Users/lizhixuan/PycharmProjects/pytorch_learning/Chapter5/sub-test/', transform=transform)

# 定义对不同数据集的读取器（loader）
# batch_size越大，每次读取的图片越多，上限是显卡的内存大小
# 在bash命令行中，使用nvidia-smi可以查看显卡当前信息
trainloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=512,
                    shuffle=True,
                    num_workers=4)
testloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=512,
                    shuffle=False,
                    num_workers=4)


# 定义测试的过程
# 训练集、验证集、测试集
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

        print('Accuracy in the test dataset: %.1f %%' % (100 * correct / total))

classes = ('cat', 'dog')

# 开始定义训练过程的各个部分
# 先定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Starting to train")
torch.set_num_threads(8)
for epoch in range(1000):

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
#         print("outputs %s  labels %s" % (outputs, labels))
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        print_gap = 10
        if i % print_gap == (print_gap-1): # 每1000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / print_gap))
            running_loss = 0.0
    # 每个epoch进行一次测试，看准确率是多少
    test()
print('Finished Training')


