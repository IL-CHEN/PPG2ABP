"""
数据切分模块
============

将 ``data.hdf5`` 中的 127260 个信号片段切分为 10 折交叉验证所需的
训练集 / 验证集 / 测试集，并保存为 pickle 文件：

- ``train{n}.p`` / ``val{n}.p`` / ``meta{n}.p``（第 n 折，n = 0~9）
- ``test.p``（独立测试集，27260 个样本）

信号在保存前已按本折的 PPG / ABP 全局极值归一化到 [0, 1] 区间；
各折的极值（``max_ppg``、``min_ppg``、``max_abp``、``min_abp``）
保存于对应 ``meta{n}.p``，供后续反归一化使用。
"""

import h5py
import numpy as np
import os
import pickle
from tqdm import tqdm

from config import (DATA_DIR, TRAIN_VAL_SAMPLES, VAL_SAMPLES_PER_FOLD,
                    VALIDATION_DATA_START, FOLD_COUNT, SIGNAL_LENGTH)


def fold_data():
    """
    将数据切分为 10 折交叉验证的训练 / 验证 / 测试集并保存。

    由于 ``data.hdf5`` 中样本已随机打乱（见 data_processing.merge_episodes），
    每折的验证集只需按预定义的起始下标截取一段连续的 10000 个样本，
    其余样本作为该折的训练集；下标 100000 之后的 27260 个样本固定为测试集。
    """
    # 打开数据文件
    fl = h5py.File(os.path.join(DATA_DIR, 'data.hdf5'), 'r')

    # 逐折切分
    for fold_id in tqdm(range(FOLD_COUNT), desc='数据切分'):

        X_train = []                        # 训练数据
        Y_train = []
        X_val = []                          # 验证数据
        Y_val = []

        # 元数据：PPG / ABP 信号的极值（用于归一化与反归一化）
        max_ppg = -10000
        min_ppg = 10000
        max_abp = -10000
        min_abp = 10000

        val_start = VALIDATION_DATA_START[fold_id]              # 验证集起始下标
        val_end = val_start + VAL_SAMPLES_PER_FOLD              # 验证集结束下标

        # 验证集之前的样本 → 训练集
        for i in tqdm(range(0, val_start), desc='训练数据第一部分'):
            X_train.append(np.array(fl['data'][i][1][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))
            Y_train.append(np.array(fl['data'][i][0][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))

            max_ppg = max(max(fl['data'][i][1]), max_ppg)       # 更新 PPG 极值
            min_ppg = min(min(fl['data'][i][1]), min_ppg)
            max_abp = max(max(fl['data'][i][0]), max_abp)       # 更新 ABP 极值
            min_abp = min(min(fl['data'][i][0]), min_abp)

        # 验证集之后的样本 → 训练集
        for i in tqdm(range(val_end, TRAIN_VAL_SAMPLES), desc='训练数据第二部分'):
            X_train.append(np.array(fl['data'][i][1][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))
            Y_train.append(np.array(fl['data'][i][0][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))

            max_ppg = max(max(fl['data'][i][1]), max_ppg)
            min_ppg = min(min(fl['data'][i][1]), min_ppg)
            max_abp = max(max(fl['data'][i][0]), max_abp)
            min_abp = min(min(fl['data'][i][0]), min_abp)

        # 验证集
        for i in tqdm(range(val_start, val_end), desc='验证数据'):
            X_val.append(np.array(fl['data'][i][1][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))
            Y_val.append(np.array(fl['data'][i][0][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))

            max_ppg = max(max(fl['data'][i][1]), max_ppg)
            min_ppg = min(min(fl['data'][i][1]), min_ppg)
            max_abp = max(max(fl['data'][i][0]), max_abp)
            min_abp = min(min(fl['data'][i][0]), min_abp)

        fl = None                           # 释放文件句柄

        # 归一化并保存训练集
        X_train = np.array(X_train)
        X_train -= min_ppg
        X_train /= (max_ppg - min_ppg)

        Y_train = np.array(Y_train)
        Y_train -= min_abp
        Y_train /= (max_abp - min_abp)

        pickle.dump({'X_train': X_train, 'Y_train': Y_train},
                    open(os.path.join(DATA_DIR, 'train{}.p'.format(fold_id)), 'wb'))

        X_train = []                        # 释放内存
        Y_train = []

        # 归一化并保存验证集
        X_val = np.array(X_val)
        X_val -= min_ppg
        X_val /= (max_ppg - min_ppg)

        Y_val = np.array(Y_val)
        Y_val -= min_abp
        Y_val /= (max_abp - min_abp)

        pickle.dump({'X_val': X_val, 'Y_val': Y_val},
                    open(os.path.join(DATA_DIR, 'val{}.p'.format(fold_id)), 'wb'))

        X_val = []                          # 释放内存
        Y_val = []

        # 保存本折元数据
        pickle.dump({'max_ppg': max_ppg, 'min_ppg': min_ppg,
                     'max_abp': max_abp, 'min_abp': min_abp},
                    open(os.path.join(DATA_DIR, 'meta{}.p'.format(fold_id)), 'wb'))

    # 测试集：下标 TRAIN_VAL_SAMPLES 之后的全部样本
    fl = h5py.File(os.path.join(DATA_DIR, 'data.hdf5'), 'r')

    X_test = []                             # 测试数据
    Y_test = []

    # 元数据沿用最后一折（此处沿用第 9 折的极值，保持与训练阶段一致）
    max_ppg = -10000
    min_ppg = 10000
    max_abp = -10000
    min_abp = 10000

    for i in tqdm(range(TRAIN_VAL_SAMPLES, len(fl['data']))):
        X_test.append(np.array(fl['data'][i][1][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))
        Y_test.append(np.array(fl['data'][i][0][:SIGNAL_LENGTH]).reshape(SIGNAL_LENGTH, 1))

        max_ppg = max(max(fl['data'][i][1]), max_ppg)
        min_ppg = min(min(fl['data'][i][1]), min_ppg)
        max_abp = max(max(fl['data'][i][0]), max_abp)
        min_abp = min(min(fl['data'][i][0]), min_abp)

    X_test = np.array(X_test)
    X_test -= min_ppg
    X_test /= (max_ppg - min_ppg)

    Y_test = np.array(Y_test)
    Y_test -= min_abp
    Y_test /= (max_abp - min_abp)

    # 保存测试集
    pickle.dump({'X_test': X_test, 'Y_test': Y_test},
                open(os.path.join(DATA_DIR, 'test.p'), 'wb'))


def main():
    fold_data()                             # 切分数据用于 10 折交叉验证


if __name__ == '__main__':
    main()
