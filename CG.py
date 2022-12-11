# Compile in 2022/11/21
import numpy as np
from PIL import Image
import os
import xlrd
from tqdm import *

rt_path = './data/'
files_list = os.listdir(rt_path)


def get_xlsx(path):
    print("LOADING...")
    workbook = xlrd.open_workbook(path)  # 打开test1.xls文件
    temp = []
    for i in tqdm(range(25)):
        sheet = workbook.sheet_by_name("{}".format(i + 1))  # 按名称读取sheet页
        rows = sheet.nrows  # 读取行数和列数
        cols = sheet.ncols
        img_slice = np.zeros((rows - 1, cols))
        for r in range(rows - 1):
            for c in range(cols):
                img_slice[r, c] = sheet.cell_value(r + 1, c)
        temp.append(img_slice)
    return np.array(temp)


file_path = r"./FindBargain/Bargain.xls"
data = get_xlsx(file_path)
total_pixel = 0
for j in tqdm(range(25)):
    im_path = rt_path + files_list[j]
    img = Image.open(im_path)
    w, h = img.size[0], img.size[1]
    total_pixel = total_pixel + w*h

print(0.8*total_pixel/1024/1024/8)
