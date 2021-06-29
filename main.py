import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):                 # 前向传播过程
        x = self.conv1(x)                  # m*1*28*28--->m*16*14*14
        x = self.conv2(x)                  # m*16*14*14--->m*32*7*7
        x = x.view(x.size(0), -1)   # flatten the output of conv2 to (batch_size, 32*7*7),m*32*7*7--->m*1568
        # x = x.view(-1, 32*7*7)
        # view函数类似reshape，返回一个新张量，其数据与self tensor相同，但形状不同
        # -1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        out = self.out(x)
        return out


if __name__ == "__main__":
    EPOCH = 1
    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MNIST = False

    train_data = torchvision.datasets.MNIST(
        root='./',
        train=True,
        transform=torchvision.transforms.ToTensor(),    # 转换PIL.Image或ndarray成tensor
        download=DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(
        root='./',
        train=False,
    )

    # 批训练50 samples,1 channel, 28*28(50,1,28,28)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # 每一步loader释放50个数据来学习
    # 为演示，先提取2000个数据
    # shape from (2000,28,28) to (2000,1,28,28),value in range(0,1),unsqueeze用于扩展维度,在指定位置(dim)加上一个维数为1的维度
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.  # type强制转换
    # torch.FloatTensor(C*H*W)，训练时normalize到[0 1]区间(原本灰度值0-255)
    test_y = test_data.targets[:2000]
    test_x = test_x.cuda()          # cuda环境
    test_y = test_y.cuda()          # cuda环境

    cnn = CNN()
    cnn = cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 单轮step总数=60000/batchsize=1200
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                test_output = cnn(test_x)
                # torch.max(input, dim, keepdim=False, *, out=None)
                # input是一个tensor，dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                pred_y = torch.max(test_output, 1)[1].detach().squeeze()
                pred_y = torch.max(test_output, 1)[1].cuda().detach().squeeze()   # cuda环境
                accuracy = float((pred_y == test_y).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| Step: ', step, '| train loss: %.4f' % loss.data,
                      '| test accuracy: %.2f' % accuracy)

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].detach().squeeze()
    pred_y = torch.max(test_output, 1)[1].cuda().detach().squeeze()      # cuda环境
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')

    # torch.save(cnn.state_dict(), './model/CNN_state.pth.tar')     # 只保存CNN参数，速度较快
    torch.save(cnn, './model/CNN.pth.tar')
