# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p16
from modulefinder import Module
import torch
from torch import nn

# nn表示神经网络模块，其中有Containers(骨架)/pooling layers(池化层)等模块
# 骨架中又有Moudle（所有神经网络的基本骨架，搭建神经网络都要继承这个类然后重写init/forward
class Tudui(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)