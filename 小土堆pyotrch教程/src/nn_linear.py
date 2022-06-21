# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p21 线性层/全连接层
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# torchvison.models中提供了许多神经网络模型

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 训练时线性层的w和b依据某种规定的分布（见文档）随机采样得到，参数为输入输出神经元个数即变量个数
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # flatten是摊平的意思，[64,3,32,32]->[1,1,1,196608]
    # 相当于torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
