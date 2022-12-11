import torch
import torchvision
from torchvision.models.detection.roi_heads import fastrcnn_loss
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from tqdm import *

# COCO数据集标签对照表
COCO_CLASSES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
          '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
          '#ffd8b1', '#e6beff', '#808080']

# 为每一个标签对应一种颜色，方便我们显示
LABEL_COLOR_MAP = {k: COLORS[i % len(COLORS)] for i, k in enumerate(COCO_CLASSES.keys())}

# 判断GPU设备是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def faster_rcnn_detection(origin_img):
    # 加载pytorch自带的预训练Faster RCNN目标检测模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    # 读取输入图像，并转化为tensor
    tf_img = TF.to_tensor(origin_img)
    tf_img = tf_img.to(device)

    # 将图像输入神经网络模型中，得到输出
    output = model(tf_img.unsqueeze(0))

    labels = output[0]['labels'].cpu().detach().numpy()  # 预测每一个obj的标签
    scores = output[0]['scores'].cpu().detach().numpy()  # 预测每一个obj的得分
    bboxes = output[0]['boxes'].cpu().detach().numpy()  # 预测每一个obj的边框
    # 这个我们只选取得分大于m的
    obj_index = np.argwhere(scores > 0.7).squeeze(axis=1).tolist()

    # 用于获取Final_data的矩形抠图
    # Rect_Matting(origin_img, bboxes, obj_index)

    # 使用ImageDraw将检测到的边框和类别打印在图片中，得到最终的输出
    # draw = ImageDraw.Draw(origin_img)
    # font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 15)
    #
    # for i in obj_index:
    #     box_location = bboxes[i].tolist()
    #     draw.rectangle(xy=box_location, outline=LABEL_COLOR_MAP[labels[i]])
    #     draw.rectangle(xy=[l + 1. for l in box_location], outline=LABEL_COLOR_MAP[labels[i]])
    #
    #     text_size = font.getsize(COCO_CLASSES[labels[i]])
    #     text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                         box_location[1]]
    #     draw.rectangle(xy=textbox_location, fill=LABEL_COLOR_MAP[labels[i]])
    #     draw.text(xy=text_location, text=COCO_CLASSES[labels[i]], fill='white', font=font)
    #
    # del draw
    #
    # origin_img.save("detect_final/result{}.png".format(m))
    return labels[obj_index]


def Rect_Matting(img, bboxes, index):
    array = np.array(img)
    array = np.transpose(array, [2, 0, 1])  # array维度 [W, H, C] -> [C, W, H]
    mask = np.zeros_like(array)
    bboxes = bboxes.astype(np.int32)

    # 掩膜1：各矩形区域置1
    # for o in index:
    #     mask[:, bboxes[o, 1]:bboxes[o, 3], bboxes[o, 0]:bboxes[o, 2]] = 1
    # 掩膜2：大矩形区域置1
    # mask[:, np.min(bboxes, axis=0)[1]:np.max(bboxes, axis=0)[3], np.min(bboxes, axis=0)[0]:np.max(bboxes, axis=0)[2]] = 1
    # 掩膜3：1加宽
    width = 30
    b = img.width
    a = img.height
    for o in index:
        if bboxes[o, 1] - width > 0:
            x1 = bboxes[o, 1] - width
        else:
            x1 = 0
        if bboxes[o, 0] - width > 0:
            y1 = bboxes[o, 0] - width
        else:
            y1 = 0
        if bboxes[o, 3] + width < a:
            x2 = bboxes[o, 3] + width
        else:
            x2 = a
        if bboxes[o, 2] + width < b:
            y2 = bboxes[o, 2] + width
        else:
            y2 = b
        mask[:, x1:x2, y1:y2] = 1
    # 法一：直接非目标区域置零
    # array = array * mask  # 点乘
    # 法二：背景模糊化
    obj = array * mask
    bac = array * (1 - mask)
    bac = np.transpose(bac, [1, 2, 0])
    bac_img = Image.fromarray(bac, mode='RGB')
    bac = np.array(bac_img.filter(ImageFilter.GaussianBlur(radius=5)))
    bac = np.transpose(bac, [2, 0, 1])
    bac = bac * (1 - mask)  # 去除边角
    array = bac + obj

    array = np.transpose(array, [1, 2, 0])
    img_final = Image.fromarray(array, mode='RGB')
    img_final.save("Rect30blur_data/" + files_list[m])


if __name__ == '__main__':
    rt_path = './Final_data/'
    files_list = os.listdir(rt_path)
    sum_pixel = 0
    for m in tqdm(range(25)):
        origin_img = Image.open(rt_path + files_list[m], mode='r').convert('RGB')
        sum_pixel = sum_pixel + origin_img.size[0] * origin_img.size[1]

        # index = faster_rcnn_detection(origin_img)
    print(sum_pixel)
