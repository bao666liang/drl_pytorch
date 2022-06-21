# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p23
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
# 误差求和或平均
loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
# 均方误差
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)

# CrossEntropyLoss输入为（N，C）格式batch_size大小和class类别数，输出为（N，*）
# *表示任意数 详见pytorch文档
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)