import numpy as np
import xlrd
import geatpy as ea
from tqdm import *


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


# 构建问题
file_path = r"./FindBargain2/Bargain.xls"
data = get_xlsx(file_path)


@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    mse_sum = 0
    img_size = 0
    size_over = 0
    for j in range(25):
        mse_sum = mse_sum + data[j, Vars[j] - 1, 1]
        img_size = img_size + data[j, Vars[j] - 1, 2]
    size_over = -min(2.3 * 1024 - img_size, 0)
    return mse_sum, size_over  # 返回目标函数值矩阵和违反约束程度矩阵


problem = ea.Problem(name='pic bargain',
                     M=1,  # 目标维数
                     maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                     Dim=25,  # 决策变量维数
                     varTypes=[1] * 25,  # 决策变量的类型列表，0：实数；1：整数
                     lb=[1] * 25,  # 决策变量下界  ,*0.05转化为缩放倍数
                     ub=[25] * 25,  # 决策变量上界
                     evalVars=evalVars)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=5000),
                                 MAXGEN=16000,  # 最大进化代数。
                                 logTras=5000,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                 trappedValue=1e-8,  # 单目标优化陷入停滞的判断阈值。
                                 maxTrappedCount=100)  # 进化停滞计数器最大上限值。
# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                  dirName='EA_result')

# Compile in 2022/9/28
