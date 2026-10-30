"""
测试集预测模块
==============

加载最优模型对（近似网络 + 精炼网络），对独立测试集 ``data/test.p``
计算预测结果并保存，避免在评估阶段重复计算：

- ``test_output_approximate.p``：近似网络输出（多输出头的第 0 项为主输出）
- ``test_output.p``：精炼网络输出（最终 ABP 波形预测）

运行前请确保已将最优模型复制为 ``models/ApproximateNetwork.h5`` 与
``models/RefinementNetwork.h5``（见 Notebook 4C 与根目录 README）。
"""

import os
import pickle

from config import (DATA_DIR, MODELS_DIR, TEST_OUTPUT_FILE,
                    TEST_OUTPUT_APPROXIMATE_FILE, SIGNAL_LENGTH)
from models import UNetDS64, MultiResUNet1D


def predict_test_data():
    """
    计算测试集预测结果并保存为 pickle 文件。
    """
    # 加载测试数据
    dt = pickle.load(open(os.path.join(DATA_DIR, 'test.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # 阶段一：近似网络预测
    mdl1 = UNetDS64(SIGNAL_LENGTH)
    mdl1.load_weights(os.path.join(MODELS_DIR, 'ApproximateNetwork.h5'))

    Y_test_pred_approximate = mdl1.predict(X_test, verbose=1)   # 近似 ABP 波形预测

    pickle.dump(Y_test_pred_approximate, open(TEST_OUTPUT_APPROXIMATE_FILE, 'wb'))

    # 阶段二：精炼网络预测（输入为近似网络的主输出头）
    mdl2 = MultiResUNet1D(SIGNAL_LENGTH)
    mdl2.load_weights(os.path.join(MODELS_DIR, 'RefinementNetwork.h5'))

    Y_test_pred = mdl2.predict(Y_test_pred_approximate[0], verbose=1)

    pickle.dump(Y_test_pred, open(TEST_OUTPUT_FILE, 'wb'))


def main():
    predict_test_data()     # 预测并保存测试集输出，避免重复计算


if __name__ == '__main__':
    main()
