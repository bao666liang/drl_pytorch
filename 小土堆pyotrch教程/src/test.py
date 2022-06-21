# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p32 利用已经训练好的模型给他提供输入(模型验证)
# Your PyTorch installation may be too old.
# 使用较高版本的pytorch训练得到的模型，在低版本的ptorch中load时，存在的版本问题
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./imgs/airplane.png"
image = Image.open(image_path)
print(image)
# image是pil.png格式，png是四通道（透明度+rgb），而vgg16网络模型输入为3*32*32
# 需要用conver保留其颜色通道，这样可以适应png,jpg等各种格式图片
# 不同截图软件保留的通道是不同的
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

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
# 加载训练好的模型
# model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))
# map是将gpu上训练的模型映射到cpu上
model = torch.load("tudui_9.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
# eval()将模型转化为测试类型
model.eval()
# no_grad不计算加载梯度，节约内存和性能
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
