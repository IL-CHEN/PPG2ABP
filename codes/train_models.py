"""
模型训练模块
============

以 10 折交叉验证方式训练 PPG2ABP 的两阶段模型：

1. ``train_approximate_network``：训练近似网络（UNetDS64，深度监督，
   损失权重 ``[1.0, 0.9, 0.8, 0.7, 0.6]``），每折训练 100 个 epoch；
2. ``train_refinement_network``：加载同折近似网络，利用其主输出头
   （``prepareDataDS``）构造精炼网络的输入，再训练精炼网络
   （MultiResUNet1D，MSE 损失）。

训练产物
--------
- ``models/UNetDS64_model1_fold{n}.h5``：近似网络权重（每折一个）
- ``models/MultiResUNet1D_model2_fold{n}.h5``：精炼网络权重（每折一个）
- ``History/UNetDS64_model1_fold{n}.p`` / ``..._model2_fold{n}.p``：训练历史

训练完成后，需按验证集表现选择最优折的模型对，并复制为
``models/ApproximateNetwork.h5`` 与 ``models/RefinementNetwork.h5``，
供预测与评估阶段使用（详见 Notebook 4C 与根目录 README）。
"""

import os
import pickle
import time
from tqdm import tqdm
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from config import (DATA_DIR, MODELS_DIR, HISTORY_DIR, EPOCHS,
                    BATCH_SIZE_APPROXIMATE, BATCH_SIZE_REFINEMENT,
                    DEEP_SUPERVISION_LOSS_WEIGHTS, GPU_COOLDOWN_SECONDS,
                    APPROXIMATE_NETWORK_NAME, REFINEMENT_NETWORK_NAME,
                    SIGNAL_LENGTH)
from models import MODEL_REGISTRY
from helper_functions import prepareLabel, prepareDataDS


def _load_fold_data(foldname):
    """
    加载指定折的训练 / 验证数据与元数据。

    参数:
        foldname: 折编号（0~9）

    返回:
        tuple: (X_train, Y_train, X_val, Y_val, max_ppg, min_ppg, max_abp, min_abp)
    """
    dt = pickle.load(open(os.path.join(DATA_DIR, 'train{}.p'.format(foldname)), 'rb'))
    X_train = dt['X_train']
    Y_train = dt['Y_train']

    dt = pickle.load(open(os.path.join(DATA_DIR, 'val{}.p'.format(foldname)), 'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']

    dt = pickle.load(open(os.path.join(DATA_DIR, 'meta{}.p'.format(foldname)), 'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    return X_train, Y_train, X_val, Y_val, max_ppg, min_ppg, max_abp, min_abp


def _prepare_deep_supervision(Y_train, Y_val):
    """
    为深度监督训练准备标签（近似网络阶段使用）。

    返回:
        tuple: (训练标签字典, 验证标签字典)
    """
    Y_train = prepareLabel(Y_train)
    Y_val = prepareLabel(Y_val)
    return Y_train, Y_val


def train_approximate_network():
    """
    以 10 折交叉验证方式训练近似网络（UNetDS64，深度监督）。
    """
    # 创建保存目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # 近似网络：UNetDS64；精炼网络：MultiResUNet1D
    mdlName1 = APPROXIMATE_NETWORK_NAME
    mdlName2 = REFINEMENT_NETWORK_NAME

    # 10 折交叉验证
    for foldname in tqdm(range(10), desc='近似网络 10 折训练'):

        print('----------------')
        print('开始训练第 {} 折'.format(foldname + 1))
        print('----------------')

        X_train, Y_train, X_val, Y_val, _, _, _, _ = _load_fold_data(foldname)

        # 为深度监督训练准备五层标签
        Y_train, Y_val = _prepare_deep_supervision(Y_train, Y_val)

        # 创建近似网络
        mdl1 = MODEL_REGISTRY[mdlName1](SIGNAL_LENGTH)

        # 损失：MAE；优化器：Adam；监控指标：MSE；深度监督各头损失权重
        mdl1.compile(loss='mean_absolute_error', optimizer='adam',
                     metrics=['mean_squared_error'],
                     loss_weights=DEEP_SUPERVISION_LOSS_WEIGHTS)

        # 依据验证集主输出头损失保存最优权重
        checkpoint1_ = ModelCheckpoint(
            os.path.join(MODELS_DIR, '{}_model1_fold{}.h5'.format(mdlName1, foldname)),
            verbose=1, monitor='val_out_loss', save_best_only=True, mode='auto')

        # 训练近似网络（100 个 epoch）
        history1 = mdl1.fit(
            X_train,
            {'out': Y_train['out'], 'level1': Y_train['level1'],
             'level2': Y_train['level2'], 'level3': Y_train['level3'],
             'level4': Y_train['level4']},
            epochs=EPOCHS, batch_size=BATCH_SIZE_APPROXIMATE,
            validation_data=(X_val,
                             {'out': Y_val['out'], 'level1': Y_val['level1'],
                              'level2': Y_val['level2'], 'level3': Y_val['level3'],
                              'level4': Y_val['level4']}),
            callbacks=[checkpoint1_], verbose=1)

        # 保存训练历史
        pickle.dump(history1, open(
            os.path.join(HISTORY_DIR, '{}_model1_fold{}.p'.format(mdlName1, foldname)), 'wb'))

        mdl1 = None                         # 释放显存

        time.sleep(GPU_COOLDOWN_SECONDS)    # 等待 GPU 显存释放


def train_refinement_network():
    """
    以 10 折交叉验证方式训练精炼网络（MultiResUNet1D）。

    每折先加载同折近似网络，用其主输出头对训练 / 验证数据做预测，
    以预测结果作为精炼网络的输入进行第二阶段训练。
    """
    mdlName1 = APPROXIMATE_NETWORK_NAME    # 近似网络
    mdlName2 = REFINEMENT_NETWORK_NAME     # 精炼网络

    # 10 折交叉验证
    for foldname in tqdm(range(10), desc='精炼网络 10 折训练'):

        print('----------------')
        print('开始训练第 {} 折'.format(foldname + 1))
        print('----------------')

        X_train, Y_train, X_val, Y_val, _, _, _, _ = _load_fold_data(foldname)

        # 准备深度监督标签（仅使用主输出头 out）
        Y_train, Y_val = _prepare_deep_supervision(Y_train, Y_val)

        # 加载同折近似网络并生成第二阶段输入
        mdl1 = MODEL_REGISTRY[mdlName1](SIGNAL_LENGTH)
        mdl1.load_weights(os.path.join(MODELS_DIR, '{}_model1_fold{}.h5'.format(mdlName1, foldname)))

        X_train = prepareDataDS(mdl1, X_train)   # 训练输入：近似网络主输出
        X_val = prepareDataDS(mdl1, X_val)       # 验证输入：近似网络主输出

        mdl1 = None                              # 释放显存

        # 创建精炼网络
        mdl2 = MODEL_REGISTRY[mdlName2](SIGNAL_LENGTH)

        # 损失：MSE；优化器：Adam；监控指标：MAE
        mdl2.compile(loss='mean_squared_error', optimizer='adam',
                     metrics=['mean_absolute_error'])

        # 依据验证集损失保存最优权重
        checkpoint2_ = ModelCheckpoint(
            os.path.join(MODELS_DIR, '{}_model2_fold{}.h5'.format(mdlName2, foldname)),
            verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

        # 训练精炼网络（100 个 epoch）
        history2 = mdl2.fit(X_train, Y_train['out'], epochs=EPOCHS,
                            batch_size=BATCH_SIZE_REFINEMENT,
                            validation_data=(X_val, Y_val['out']),
                            callbacks=[checkpoint2_])

        # 保存训练历史
        pickle.dump(history2, open(
            os.path.join(HISTORY_DIR, '{}_model2_fold{}.p'.format(mdlName2, foldname)), 'wb'))

        time.sleep(GPU_COOLDOWN_SECONDS)    # 等待 GPU 显存释放


def main():
    train_approximate_network()     # 10 折训练近似网络
    train_refinement_network()      # 10 折训练精炼网络


if __name__ == '__main__':
    main()
