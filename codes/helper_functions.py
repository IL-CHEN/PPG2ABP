"""
辅助函数模块
============

为两阶段训练流水线提供数据与标签预处理工具：

- ``prepareData``：利用已有模型生成第二阶段训练的输入（两阶段流水线）。
- ``prepareDataDS``：深度监督流水线中，利用近似网络生成精炼网络输入。
- ``prepareLabel``：将原始标签降采样为深度监督所需的五层标签
  （out / level1 ~ level4，逐级 2 倍下采样）。
"""

import numpy as np
from tqdm import tqdm

from config import SIGNAL_LENGTH


def prepareData(mdl, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """
    为第二阶段训练准备数据（两阶段流水线）。

    利用第一阶段模型的预测输出作为第二阶段的输入特征。

    参数:
        mdl: keras 模型
        X_train: 训练输入
        X_val: 验证输入
        X_test: 测试输入
        Y_train: 训练标签
        Y_val: 验证标签
        Y_test: 测试标签

    返回:
        tuple: 第二阶段训练用的 (X_train, X_val, X_test)
    """
    X2_train = []
    X2_val = []
    X2_test = []

    YPs = mdl.predict(X_train)
    for i in tqdm(range(len(X_train))):
        X2_train.append(np.array(YPs[i]))

    YPs = mdl.predict(X_val)
    for i in tqdm(range(len(X_val))):
        X2_val.append(np.array(YPs[i]))

    YPs = mdl.predict(X_test)
    for i in tqdm(range(len(X_test))):
        X2_test.append(np.array(YPs[i]))

    X2_train = np.array(X2_train)
    X2_val = np.array(X2_val)
    X2_test = np.array(X2_test)
    return (X2_train, X2_val, X2_test)


def prepareDataDS(mdl, X):
    """
    为深度监督流水线的第二阶段训练准备数据。

    取深度监督模型的主输出头（``out``）作为第二阶段的输入特征。

    参数:
        mdl: keras 模型（深度监督模型）
        X: 输入数组（训练或验证集）

    返回:
        array: 适合第二阶段训练输入的特征
    """
    X2 = []

    YPs = mdl.predict(X)

    for i in tqdm(range(len(X)), desc='正在为深度监督准备数据'):
        X2.append(np.array(YPs[0][i]))

    return np.array(X2)


def prepareLabel(Y):
    """
    为深度监督流水线准备标签。

    将每个原始标签（长度 1024）依次用 2、4、8、16 的窗口做均值降采样，
    得到五层标签：out（原始分辨率）、level1（1/2）、level2（1/4）、
    level3（1/8）、level4（1/16）。

    返回:
        dict: 包含五层标签的字典
    """
    def approximate(inp, w_len):
        """
        对输入信号按窗口长度做均值降采样。

        参数:
            inp: 信号数组
            w_len: 窗口长度

        返回:
            array: 降采样后的信号
        """
        op = []
        for i in range(0, len(inp), w_len):
            op.append(np.mean(inp[i:i + w_len]))
        return np.array(op)

    out = {'out': [], 'level1': [], 'level2': [], 'level3': [], 'level4': []}

    for y in tqdm(Y, desc='正在准备深度监督标签'):
        # 计算各层近似（逐级 2 倍下采样）
        cA1 = approximate(np.array(y).reshape(SIGNAL_LENGTH), 2)
        cA2 = approximate(np.array(y).reshape(SIGNAL_LENGTH), 4)
        cA3 = approximate(np.array(y).reshape(SIGNAL_LENGTH), 8)
        cA4 = approximate(np.array(y).reshape(SIGNAL_LENGTH), 16)

        # 填充各层标签
        out['out'].append(np.array(y.reshape(SIGNAL_LENGTH, 1)))
        out['level1'].append(np.array(cA1.reshape(SIGNAL_LENGTH // 2, 1)))
        out['level2'].append(np.array(cA2.reshape(SIGNAL_LENGTH // 4, 1)))
        out['level3'].append(np.array(cA3.reshape(SIGNAL_LENGTH // 8, 1)))
        out['level4'].append(np.array(cA4.reshape(SIGNAL_LENGTH // 16, 1)))

    # 转换为 numpy 数组
    out['out'] = np.array(out['out'])
    out['level1'] = np.array(out['level1'])
    out['level2'] = np.array(out['level2'])
    out['level3'] = np.array(out['level3'])
    out['level4'] = np.array(out['level4'])
    return out
