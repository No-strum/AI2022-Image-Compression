import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import module
import torchvision.transforms as transforms
from tqdm import *

# 超参数
# scale_factor = 0.43
rt_path = './data/'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def img2tensor(image):
    # image = module.my_up(image, img_ori.size()[1:3])  # 高分辨视图
    image1 = np.array(image)
    image2 = np.transpose(image1, (2, 0, 1))
    image3 = torch.from_numpy(image2).float()
    return image3.unsqueeze(0)


def tensor2img(image):
    # image = module.my_up(image, img_ori.size()[1:3])  # 高分辨视图
    img1 = np.transpose(image.squeeze(0).detach().numpy(), (1, 2, 0))
    img_cut = np.clip(img1, 0, 255)
    return Image.fromarray(img_cut.astype('uint8'))


train_transform = transforms.Compose([
    transforms.ToTensor(),
])
# test
loss_func = nn.MSELoss()
file_path = r"C:/Users/tzy/Desktop/大三上/人智/作业/1/myCompress/FindBargain/"
file = open(file_path + "Label.txt", 'r')  # 打开文件
file_data = file.readlines()  # 读取所有行
label = []
total_loss = 0
for row in file_data:
    tmp_list = row.split('\t')  # 按‘ ’切分每行的数据
    tmp_list.pop()  # 去掉末尾元素
    label = [int(x) for x in tmp_list]
for i in tqdm(range(100)):
    with torch.no_grad():
        scale_factor = int(label[i])/100
        model = module.AutoEncoder()
        model.load_state_dict(torch.load("./my_para/pic{}.pth".format(i + 1)))

        im_path = rt_path + "{}.bmp".format(i + 1)
        img = Image.open(im_path)
        w, h = img.size[0], img.size[1]
        compress_img = img.resize((int(scale_factor * w), int(scale_factor * h)), Image.ANTIALIAS)
        img_out = model(train_transform(compress_img).unsqueeze(0))
        T_img = train_transform(img).unsqueeze(0)
        ori_mse = loss_func(module.my_up(img_out, T_img.size()[2:4]), T_img).item() * 3
        total_loss = total_loss + float(ori_mse)
        img_final = tensor2img(img_out*255)
        img_final.save("./low_pics/{}.jpg".format(i + 1), quality=99)
        img_final.save("./low_ppms/{}.ppm".format(i + 1))

print(total_loss)
# Compile in 2022/9/20
