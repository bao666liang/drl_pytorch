# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念 p25 数据集：CIFAR10/ImageNet 网络模型：vgg16 在torchvision中
# 现在imagenet不再可以公开访问，必须手动下载到根目录100G
import torchvision

# train_data = torchvision.datasets.ImageNet("./data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 因为CIFAR10分类为10，vgg16输出为1000/4096，因此需要加全连接层修改输出为10，其是在模型的class模块中添加
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


