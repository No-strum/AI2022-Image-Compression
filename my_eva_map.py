# 计算mAP指标
def get_mAP(y_trues, y_preds, m, n, digts):
    """
    :param y_trues:1-D list
    :param y_preds:2-D list
    :param m:number of labels of each sample
    :param n: number of samples
    :param digts: decimal places
    :return: a float number
    """
    k = len(y_preds[0])  # 每个测试样本的预测标签数量
    y_trues = [[y_trues[i]] * k for i in range(len(y_trues))]
    avg_precisions = []  # 存放每个样本的average precision的列表
    for j in range(len(y_trues)):
        y_true = y_trues[j]  # 举例：真实标签列表，[1, 1, 1]
        y_pred = y_preds[j]  # 举例：预测标签列表，[0, 0, 1]
        cur_hit_num = 0  # 当前命中数量
        hit_list = [int(y_true[s] == y_pred[s]) for s in range(k)]  # 举例：[0, 1, 1]
        print('hit_list:', hit_list)
        precision_k = []
        k_index = 0  # 元素对应索引
        for o in hit_list:
            k_index += 1
            if o == 1:
                cur_hit_num += 1  # 遍历到目前为止命中元素数量
                # precision@k
                precision_k.append(float(cur_hit_num) / k_index)
            else:
                cur_hit_num += 0
                # precision@k
                precision_k.append(float(cur_hit_num) / k_index)
        print('precision_k:', precision_k)

        try:
            avg_precision = (1 / m) * sum([hit_list[s] * precision_k[s] for s in range(k)])
        except ZeroDivisionError:
            avg_precision = 0.0
        print('avg_precision:', avg_precision)
        avg_precisions.append(avg_precision)
    print('avg_precisions:', avg_precisions)
    mAP = round(sum(avg_precisions) / n, digts)
    return mAP


if __name__ == '__main__':
    # 注意输入格式
    y_tures = [1, 2, 3]
    y_preds = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    map = get_mAP(y_tures, y_preds, m=1, n=3, digts=3)
    print('mAP:', map)

# Compile in 2022/12/10
