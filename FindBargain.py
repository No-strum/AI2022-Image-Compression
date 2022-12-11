import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import os
import xlwt
import my_eva
from tqdm import *
from sklearn.metrics import recall_score
from skimage.metrics import structural_similarity as ssim

rt_path = './data/'
add_path = 'FindBargain2/'
files_list = os.listdir(rt_path)

book = xlwt.Workbook(encoding='utf-8', style_compression=0)

# test
loss_func = nn.MSELoss(reduction="mean")
for i in tqdm(range(25)):
    locals()["sheet{}".format(i + 1)] = book.add_sheet('{}'.format(i + 1), cell_overwrite_ok=True)
    im_path = rt_path + files_list[i]
    img = Image.open(im_path).convert("RGB")
    w, h = img.size[0], img.size[1]
    index_ori = my_eva.faster_rcnn_detection(img)
    for m in range(1, 26):
        col = ('压缩比例', 'mAP', '所占空间')
        for j in range(3):
            locals()["sheet{}".format(i + 1)].write(0, j, col[j])

        scale_factor = m / 25
        locals()["sheet{}".format(i + 1)].write(m, 0, scale_factor)

        scale_path = "C:/Users/tzy/Desktop/大三上/人智/期末/MyCompressV2/" + add_path + "{}/".format(scale_factor)

        if not os.path.exists(scale_path):
            os.mkdir(scale_path)
        if os.path.exists(scale_path + files_list[i]):
            locals()["sheet{}".format(i + 1)].write(m, 2, os.path.getsize(scale_path + files_list[i]) / 1024)

        compress_img = img.resize((int(scale_factor * w), int(scale_factor * h)), Image.ANTIALIAS)
        up_img = compress_img.resize((w, h), Image.BICUBIC)
        index_now = my_eva.faster_rcnn_detection(up_img)
        if index_now.size < index_ori.size:
            index_nowX = np.append(index_now, np.zeros(index_ori.size - index_now.size))
            index_oriX = index_ori
        else:
            index_nowX = index_now
            index_oriX = np.append(index_ori, np.zeros(index_now.size - index_ori.size))

        # imgX = np.array(img)
        # up_imgX = np.array(up_img)
        # my_SSIM = ssim(imgX, up_imgX, multichannel=True)
        RC = recall_score(index_nowX, index_oriX, average="micro")
        # mAP = len([x for x in index_now if x in index_ori]) / (len(index_ori) + 1)  # 先用TP代替mAP

        locals()["sheet{}".format(i + 1)].write(m, 1, float(RC))
        # compress_img.save("C:/Users/tzy/Desktop/大三上/人智/期末/MyCompressV2/" + add_path + '{}.ppm'.format(m))
        # cmd = "jpeg.exe -ls 2 -c C:/Users/tzy/Desktop/大三上/人智/期末/MyCompressV2/" + add_path + "{}.ppm ".format(m) \
        #       + "C:/Users/tzy/Desktop/大三上/人智/期末/MyCompressV2/" + add_path + "{}/".format(scale_factor) + files_list[i]
        # os.system(cmd)

save_path = 'C:/Users/tzy/Desktop/大三上/人智/期末/MyCompressV2/' + add_path + 'Bargain.xls'
book.save(save_path)
# FindBargain~Rec15数据 FindBargain2~前后景分割后数据
# Compile in 2022/9/20
