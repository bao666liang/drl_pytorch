# p6和p7

from cProfile import label
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image #读取图片
import os #获取图片地址
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs") #将对应的事件文件存储到logs文件夹下

# Dataset抽象类提供数据地址和数据大小，要重写

class MyData(Dataset):
    # __init__一般用来创建类的全局变量self.给后面使用
    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir) #将两个路径名拼接，避免win和linux不同//和/
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path) #将文件夹下的文件名变成列表，内容为文件名即图片名
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()
    # 读取其中每一个图片及label
    def __getitem__(self, idx):
        img_name = self.image_list[idx]  #self.意味用的是全局变量
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'r') as f:
            label = f.readline()

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample
    # 整个数据集有多少个
    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # 创建图片label和内容有多种方式，如图片名为label或图片文件夹名为label(二分类)或txt文件一一对应
    root_dir = "/home/wanbaoliang/pytorch-tutorial/src/dataset/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset
    print(len(train_dataset))
    img, label = train_dataset[3]
    # 直接img.show()不行？

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    writer.add_image('error', train_dataset[119]['img'])
    
    writer.close()
    # for i, j in enumerate(dataloader):
    #     # imgs, labels = j
    #     print(type(j))
    #     print(i, j['img'].shape)
    #     # writer.add_image("train_data_b2", make_grid(j['img']), i)
    #
    # writer.close()



