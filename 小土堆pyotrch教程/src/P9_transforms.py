
# p12/p13 常见的Ttansforms

from prometheus_client import Summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms # 图片变换,里面有各种图片变换工具Class

# 输入：PIL python自带 Image.open()
# 输出：tensor  Totensor()
# 作用：narrarys  cv.imread()
class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.readline()

        if self.transform:
            img = transform(img)


        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

# 转换tensor和归一化
# writer = SummaryWriter("logs") # 日志文件存放位置
# img = Image.open("dirpath")
# print(img)
# trans_tensor = transforms.ToTensor()
# img_tensor = trans_tensor(img)
# writer.add_image("Totemsor", img_tensor)
# trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) rgb每个通道都给平均值和方差来归一化 （x-u）/d
# img_morm = trans_norm(img_tensor)
# print(img_morm[0][0][0])
# writer.add_image("norm", img_morm, 1)
# writer.close()

transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
root_dir = "./dataset/train"
image_ants = "ants_image"
label_ants = "ants_label"
ants_dataset = MyData(root_dir, image_ants, label_ants, transform=transform)
image_bees = "bees_image"
label_bees = "bees_label"
bees_dataset = MyData(root_dir, image_bees, label_bees, transform=transform)




