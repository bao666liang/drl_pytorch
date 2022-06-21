# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p26
import torch
from model_save import *
import torchvision
from torch import nn
# 保存方式1的加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2的加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1 以方式一形式读取自定义模型时要先将该模型复制或引用到读取文件中(即下文)否则会报错，但不必加
# tudui = Tudui() 了，这是因为以防引用了别的模型 ，但可以用这个方法：from model_save import *就不必复制了
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('tudui_method1.pth')
print(model)