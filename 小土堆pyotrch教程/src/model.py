# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p22 搭建CIFAR10网络模型，根据公式计算padding填充参数（卷积后尺寸不变）
import torch
from torch import nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    # 下面是为了对网络结构参数检验（输出size是否正确）
    # 64*10是64张图片，10个概率值（分类）
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)