import os
from PIL import Image
import numpy as np
import xlrd
import torch.nn as nn
import torch

file_path = r"C:/Users/tzy/Desktop/大三上/人智/期末/myCompressV2/FindBargain2/"
workbook = xlrd.open_workbook(file_path + 'Label.xls')
sheet = workbook.sheet_by_name('1')
label = []
for i in range(25):
    label.append(int(sheet.cell_value(0, i)))
# label = [8] * 100
# label = np.array(label)
rt_path = './data/'
files_list = os.listdir(rt_path)
for i in range(25):
    cmd = "copy C:\\Users\\tzy\\Desktop\\大三上\\人智\\期末\\myCompressV2\\FindBargain2\\{}\\".format(
        label[i] / 25) + files_list[
              i] + " C:\\Users\\tzy\\Desktop\\大三上\\人智\\期末\\myCompressV2\\FindBargain2\\Final\\"
    os.system(cmd)

# 下面为测试jpg对不对的代码
# def img2tensor(image):
#     # image = module.my_up(image, img_ori.size()[1:3])  # 高分辨视图
#     image1 = np.array(image)
#     image2 = np.transpose(image1, (2, 0, 1))
#     image3 = torch.from_numpy(image2).float()
#     return image3.unsqueeze(0)
#
#
# loss_func = nn.MSELoss(reduction="mean")
# temp = 0
# for j in range(100):
#     img = Image.open(file_path + "Final/" + "{}.ppm".format(j + 1))
#     w, h = img.size[0], img.size[1]
#     up_img = module.my_up(img2tensor(img), (720, 1280))
#     img_ori = Image.open('./data/{}.bmp'.format(j + 1))
#     ori_mse = loss_func(up_img, img2tensor(img_ori)).item() * 3
#     temp = temp + module.PSNR(ori_mse)
# print(temp)

# Compile in 2022/9/28
