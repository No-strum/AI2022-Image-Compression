import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import os
from PIL import Image
import module
import string

# 超参数
EPOCH = 3000
BATCH_SIZE = 1
LR = 0.003
rt_path = './data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.ToTensor(),
])


def img2tensor(image):
    # image = module.my_up(image, img_ori.size()[1:3])  # 高分辨视图
    image1 = np.array(image)
    image2 = np.transpose(image1, (2, 0, 1))
    image3 = torch.from_numpy(image2).float()
    return image3.unsqueeze(0)


class PICDataset(Data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img = self.data_info[index]
        imgs = Image.open(path_img)  # 0~255
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, _, pic_names in os.walk(data_dir):
            # 遍历图片
            for j in range(len(pic_names)):
                img_name = pic_names[j]
                path_img = os.path.join(root, img_name)
                data_info.append(path_img)

        return data_info


train_data = PICDataset(data_dir='./data', transform=train_transform)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

file_path = r"C:/Users/tzy/Desktop/大三上/人智/作业/1/myCompress/FindBargain/"
file = open(file_path + "Label.txt", 'r')  # 打开文件
file_data = file.readlines()  # 读取所有行
label = []
for row in file_data:
    tmp_list = row.split('\t')  # 按‘ ’切分每行的数据
    tmp_list.pop()  # 去掉末尾元素
    label = [int(x) for x in tmp_list]

for num, data in enumerate(train_loader):
    # 定义一个编码器对象
    print("pic{}".format(num))
    autoencoder = module.AutoEncoder().to(device)
    # if os.path.exists("./my_para/pic{}.pth".format(num + 1)):
    #     autoencoder.load_state_dict(torch.load("./my_para/pic{}.pth".format(num + 1)))
    # 训练编码器
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    ori = data
    ori = ori.to(device)
    img = Image.open(rt_path + "{}.bmp".format(num + 1))
    w, h = img.size[0], img.size[1]
    scale_factor = label[num] / 100
    compress = img.resize((int(scale_factor * w), int(scale_factor * h)), Image.ANTIALIAS)
    total_ori = 0
    total_net = 0

    for epoch in range(EPOCH + 1):
        img_size = ori.size()[2:4]
        compress_T = train_transform(compress).to(device)
        decoded_x = autoencoder(compress_T.unsqueeze(0))
        img_high = module.my_up(decoded_x, img_size)
        loss1 = loss_func(img_high[:, 0], ori[:, 0])
        loss2 = loss_func(img_high[:, 1], ori[:, 1])
        loss3 = loss_func(img_high[:, 2], ori[:, 2])
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        net_mse = loss.item()
        if epoch % 1500 == 0:
            print("Epoch {}".format(epoch))
            print("net lose:", net_mse)
        if epoch % EPOCH == 0:
            ori_mse = loss_func(module.my_up(compress_T.unsqueeze(0), img_size),
                                ori).item() * 3  # 与OPT2的转化公式为*255*255/3
            print("ori lose:", ori_mse)
            total_ori = total_ori + float(ori_mse)
            total_net = total_net + float(net_mse)
        optimizer.step()  # apply gradients

    torch.save(autoencoder.state_dict(), "./my_para/pic{}.pth".format(num + 1))
print("****************")
