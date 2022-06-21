import torchvision

# 准备的测试数据集  p15
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# dataloader是从datasets每次取多少数据并打包和取的方式
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target，因为Dataloader中getitem返回img和target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2): # 因为shuffle=true所以两次epoch中相同step下不同
    step = 0
    for data in test_loader:
        imgs, targets = data
        # imgs直接作为神经网络的输入
        # print(imgs.shape) 每次都64张图片打包（64*3*32*32）
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step) #不是image而是images
        step = step + 1

writer.close()


