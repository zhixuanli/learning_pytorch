from __future__ import print_function
import torch as t
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
# print(net)

input = Variable(t.randn(1, 1, 32, 32))
# print(input)

target = Variable(t.Tensor(1, 10))
for i in range(10):
    target[0][i] = i

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)  # 0.001 is better than 0.01

for epoch in range(100):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    print("epoch: %d loss: %d" % (epoch, loss))
    loss.backward()
    optimizer.step()

    # print("epoch: %d loss: %d" % (epoch, loss))




