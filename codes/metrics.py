"""
评估指标模块
============

实现了论文评估阶段使用的各类指标。目前包含 BHS 标准（英国高血压学会
评估协议）：统计预测误差落在 ≤5、≤10、≤15 mmHg 三个区间内的样本占比。
"""

import numpy as np


def BHS_standard(err):
    """
    计算 BHS 标准指标。

    参数:
        err: 绝对误差数组（mmHg）

    返回:
        tuple: 误差 ≤5 mmHg、≤10 mmHg、≤15 mmHg 的样本占比（百分比）
    """
    abs_err = np.abs(np.asarray(err, dtype=float))

    leq5 = np.sum(abs_err <= 5)
    leq10 = np.sum(abs_err <= 10)
    leq15 = np.sum(abs_err <= 15)

    total = len(abs_err)
    return (leq5 * 100.0 / total, leq10 * 100.0 / total, leq15 * 100.0 / total)
