# p14 torchvison中的标准数据集使用  CIFAR10

import tensorboard
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 数据集可以自己下载复制到dataset中，download设置为true会自动校验数据集是否存在然后解压，与CFAR10可以不同
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

print(test_set[0]) # 查看测试集的第一个
print(test_set.classes) # class为label 10

# img, target = test_set[0]
# print(img)
# print(target) # 3 用数字代表label
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("p10") # 日志文件存放位置
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i) # 添加到tensorboard

writer.close()