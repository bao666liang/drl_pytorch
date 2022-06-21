from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# p8 tensorboard(2) ：向其中写一些图像 writer.add_image()
# 深度神经网络就像一个黑盒子，其内部的组织、结构、以及其训练过程很难理清楚，这给深度神经网络原理的理解和工程化带来了很大的挑战。
# Tensorboard是tensorflow内置的一个可视化工具，它通过将tensorflow程序输出的日志文件的信息可视化使得tensorflow程序的理解、调试和优化更加简单高效。
writer = SummaryWriter("logs") 
image_path = "/home/wanbaoliang/pytorch-tutorial/src/dataset/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(image_path) # PIL读取的是PIL.Jpeg类型，要进行转换再读取训练 opencv读取的是numpy型 pip install opencv-python
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
# tensorboard启动命令$:tensorboard --logdir=logs --port=6007 避免都打开同一个端口号6006 logdir要赋事件文件夹名，命令要在logs上级目录
writer.add_image("train", img_array, 1, dataformats='HWC')
# def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
# img_tensor:图像数据类型 torch.Tensor, numpy.array, or string/blobname
# step:对图像分类来说一个step就是一张图片（同一个tag下）
# dataformats:指定图片表达格式 通道数C 宽度W 高度H


for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
    # writer.add_scalar()
    # def add_scalar(self, tag, scalar_value, global_step=None, walltime=None)
    # tag:图表标题
    # scalar_value:需要保存的数值 y轴
    # step:训练步数 x轴
    # walltime:可选参数，一般不常用，为默认

writer.close() #写入后要记得写关闭