"""
评估模块
========

基于测试集预测结果（``test_output.p``）对 PPG2ABP 进行多维度评估：

- ``predicting_ABP_waveform``：交互式可视化单条测试样本的波形预测；
- ``evaluate_BHS_Standard``：BHS 标准评估（英国高血压学会协议）；
- ``evaluate_AAMI_Standard``：AAMI 标准评估（美国医疗器械促进协会协议）；
- ``evaluate_BP_Classification``：血压分级（正常 / 高血压前期 / 高血压）准确率；
- ``bland_altman_plot``：Bland-Altman 一致性分析图；
- ``regression_plot``：目标值与估计值的回归图及统计量（r 值、p 值）。

所有评估函数均依赖 ``test_output.p`` 文件，请先运行 ``predict_test.py``。
"""

import os
import pickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import (DATA_DIR, TEST_OUTPUT_FILE, TEST_OUTPUT_APPROXIMATE_FILE,
                    SAMPLING_FREQUENCY, SIGNAL_LENGTH, TEST_SAMPLES,
                    DBP_CLASS_BOUNDS, SBP_CLASS_BOUNDS)
from metrics import BHS_standard

sns.set()

# 血压分级类别（顺序同时用于混淆矩阵坐标轴）
BP_CLASSES = ['高血压', '正常血压', '高血压前期']


def _load_test_data():
    """加载测试集、元数据与预计算预测结果（所有评估函数共用）。"""
    dt = pickle.load(open(os.path.join(DATA_DIR, 'test.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    dt = pickle.load(open(os.path.join(DATA_DIR, 'meta9.p'), 'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_test_pred_approximate = pickle.load(open(TEST_OUTPUT_APPROXIMATE_FILE, 'rb'))
    Y_test_pred = pickle.load(open(TEST_OUTPUT_FILE, 'rb'))

    return X_test, Y_test, max_ppg, min_ppg, max_abp, min_abp, \
        Y_test_pred_approximate, Y_test_pred


def _draw_histogram_panel(ax, errors, title, xlim, ytick_labels,
                          draw_thresholds=True, draw_xticks=True):
    """
    绘制误差直方图（左侧为样本数、右侧为百分比）。

    参数:
        ax: 绘图坐标轴（主坐标轴）
        errors: 误差数组（mmHg）
        title: 子图标题
        xlim: 横轴范围 (xmin, xmax)
        ytick_labels: 右侧百分比轴刻度标签（逐面板固定刻度）
        draw_thresholds: 是否绘制 BHS 阈值参考线
        draw_xticks: 是否按 5 mmHg 步长设置横轴刻度
    """
    ax2 = ax.twinx()
    sns.distplot(errors, bins=100, kde=False, rug=False, ax=ax)
    sns.distplot(errors, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(ytick_labels)
    ax.set_xlabel('误差 (mmHg)', fontsize=14)
    ax.set_ylabel('样本数', fontsize=14)
    ax2.set_ylabel('样本百分比', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    if draw_xticks:
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 5))
    if draw_thresholds:
        # BHS 阈值参考线
        newline([5, 0], [5, 10000])
        newline([10, 0], [10, 10000])
        newline([15, 0], [15, 10000])
    plt.tight_layout()


def newline(p1, p2):
    """
    在坐标轴上绘制两点连线（辅助线）。

    参数:
        p1: 第一个点的坐标
        p2: 第二个点的坐标

    返回:
        mlines.Line2D: 绘制的线段
    """
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax], linewidth=1, linestyle='--')
    ax.add_line(l)
    return l


def predicting_ABP_waveform():
    """
    交互式可视化：从测试集中选择一条样本，绘制输入 PPG 波形以及
    近似网络 / 精炼网络的 ABP 预测与真实值对比。

    输入 1~27260 选择样本，输入 -1 退出。
    """
    X_test, Y_test, max_ppg, min_ppg, max_abp, min_abp, \
        Y_test_pred_approximate, Y_test_pred = _load_test_data()

    # 近似网络输出为多输出头，取第 0 项（主输出）
    Y_test_pred_approximate = Y_test_pred_approximate[0]

    while True:      # 交互式命令行
        indix = int(input(
            '请输入 1~{} 之间的记录编号进行可视化（输入 -1 退出）。\n> '.format(TEST_SAMPLES)))

        if indix == -1:             # 退出
            break

        if indix < 1 or indix > TEST_SAMPLES:   # 无效编号
            continue

        indix -= 1

        # 反归一化还原真实量纲（mmHg）
        ppg_signal = X_test[indix] * max_ppg + min_ppg
        abp_signal_pred_approximate = Y_test_pred_approximate[indix] * max_abp + min_abp
        abp_signal_pred = Y_test_pred[indix] * max_abp + min_abp
        abp_signal_ground_truth = Y_test[indix] * max_abp + min_abp

        # 时间轴（秒）
        duration = len(ppg_signal) / SAMPLING_FREQUENCY
        time_scale = np.arange(0, duration, duration / len(ppg_signal))

        plt.figure(figsize=(30, 15))

        plt.subplot(5, 1, 1)
        plt.plot(time_scale, ppg_signal, c='k', linewidth=2)
        plt.title('输入 PPG 信号', fontsize=20)

        plt.subplot(5, 1, 2)
        plt.plot(time_scale, abp_signal_pred_approximate, c='r', linewidth=2)
        plt.ylabel('ABP (mmHg)', fontsize=15)
        plt.title('近似网络输出', fontsize=20)

        plt.subplot(5, 1, 3)
        plt.plot(time_scale, abp_signal_pred, c='b', linewidth=2)
        plt.ylabel('ABP (mmHg)', fontsize=15)
        plt.title('精炼网络输出', fontsize=20)

        plt.subplot(5, 1, 4)
        plt.plot(time_scale, abp_signal_ground_truth, c='g', linewidth=2)
        plt.ylabel('ABP (mmHg)', fontsize=15)
        plt.title('真实 ABP 波形', fontsize=20)

        plt.subplot(5, 1, 5)
        plt.plot(time_scale, abp_signal_pred_approximate, c='r',
                 label='近似网络输出', linewidth=2)
        plt.plot(time_scale, abp_signal_pred, c='b',
                 label='精炼网络输出', linewidth=2)
        plt.plot(time_scale, abp_signal_ground_truth, c='g',
                 label='真实 ABP 波形', linewidth=2)
        plt.title('对比', fontsize=20)
        plt.xlabel('时间 (秒)', fontsize=15)
        plt.ylabel('ABP (mmHg)', fontsize=15)

        plt.tight_layout()
        plt.show()


def _calc_error(Ytrue, Ypred, max_abp, signed=False):
    """
    计算 SBP / DBP / MAP 的误差。

    参数:
        Ytrue: 真实波形数组
        Ypred: 预测波形数组
        max_abp: ABP 信号最大值（归一化还原系数）
        signed: True 时返回带符号误差（AAMI 标准），否则返回绝对误差（BHS 标准）

    返回:
        tuple: (SBP 误差, DBP 误差, MAP 误差)
    """
    sbps = []
    dbps = []
    maps = []

    for i in range(len(Ytrue)):
        y_t = Ytrue[i].ravel()
        y_p = Ypred[i].ravel()

        if signed:
            dbps.append(max_abp * (min(y_p) - min(y_t)))
            sbps.append(max_abp * (max(y_p) - max(y_t)))
            maps.append(max_abp * (np.mean(y_p) - np.mean(y_t)))
        else:
            dbps.append(max_abp * abs(min(y_t) - min(y_p)))
            sbps.append(max_abp * abs(max(y_t) - max(y_p)))
            maps.append(max_abp * abs(np.mean(y_t) - np.mean(y_p)))

    return sbps, dbps, maps


def evaluate_BHS_Standard():
    """
    依据 BHS 标准评估 PPG2ABP：统计 SBP / DBP / MAP 预测误差
    落在 ≤5、≤10、≤15 mmHg 区间内的样本占比，并绘制误差分布图。
    """
    _, Y_test, _, _, max_abp, _, _, Y_pred = _load_test_data()

    sbps, dbps, maps = _calc_error(Y_test, Y_pred, max_abp, signed=False)

    # 计算 BHS 标准指标
    sbp_percent = BHS_standard(sbps)
    dbp_percent = BHS_standard(dbps)
    map_percent = BHS_standard(maps)

    print('----------------------------')
    print('|        BHS 标准指标        |')
    print('----------------------------')
    print('----------------------------------------')
    print('|     | <=5mmHg | <=10mmHg | <=15mmHg |')
    print('----------------------------------------')
    print('| DBP | {} %  | {} %  | {} %  |'.format(
        round(dbp_percent[0], 1), round(dbp_percent[1], 1), round(dbp_percent[2], 1)))
    print('| MAP | {} %  | {} %  | {} %  |'.format(
        round(map_percent[0], 1), round(map_percent[1], 1), round(map_percent[2], 1)))
    print('| SBP | {} %  | {} %  | {} %  |'.format(
        round(sbp_percent[0], 1), round(sbp_percent[1], 1), round(sbp_percent[2], 1)))
    print('----------------------------------------')

    # 绘制误差分布图
    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 3, 1)
    _draw_histogram_panel(ax1, sbps, '收缩压 SBP 预测绝对误差', (0, 60),
                          ['0 %', '3.67 %', '7.34 %', '11.01 %',
                           '14.67 %', '18.34 %', '22.01 %'])

    ax1 = plt.subplot(1, 3, 2)
    _draw_histogram_panel(ax1, dbps, '舒张压 DBP 预测绝对误差', (0, 60),
                          ['0 %', '7.34 %', '14.67 %', '22.01 %',
                           '29.35 %', '36.68 %', '44.02 %'])

    ax1 = plt.subplot(1, 3, 3)
    _draw_histogram_panel(ax1, maps, '平均动脉压 MAP 预测绝对误差', (0, 60),
                          ['0 %', '7.34 %', '14.67 %', '22.01 %',
                           '29.35 %', '36.68 %', '44.02 %', '51.36 %'])

    plt.show()


def evaluate_AAMI_Standard():
    """
    依据 AAMI 标准评估 PPG2ABP：计算 SBP / DBP / MAP 预测误差的
    均值（ME）与标准差（STD），并绘制误差分布图。
    """
    _, Y_test, _, _, max_abp, _, _, Y_pred = _load_test_data()

    sbps, dbps, maps = _calc_error(Y_test, Y_pred, max_abp, signed=True)

    print('---------------------')
    print('|   AAMI 标准指标    |')
    print('---------------------')
    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(round(np.mean(dbps), 3), round(np.std(dbps), 3)))
    print('| MAP | {} | {} |'.format(round(np.mean(maps), 3), round(np.std(maps), 3)))
    print('| SBP | {} | {} |'.format(round(np.mean(sbps), 3), round(np.std(sbps), 3)))
    print('-----------------------')

    # 绘制误差分布图
    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 3, 1)
    _draw_histogram_panel(ax1, dbps, '舒张压 DBP 预测误差', (-50, 50),
                          ['0 %', '7.34 %', '14.67 %', '22.01 %',
                           '29.35 %', '36.68 %', '44.02 %'],
                          draw_thresholds=False, draw_xticks=False)

    ax1 = plt.subplot(1, 3, 2)
    _draw_histogram_panel(ax1, maps, '平均动脉压 MAP 预测误差', (-50, 50),
                          ['0 %', '7.34 %', '14.67 %', '22.01 %',
                           '29.35 %', '36.68 %', '44.02 %', '51.36 %'],
                          draw_thresholds=False, draw_xticks=False)

    ax1 = plt.subplot(1, 3, 3)
    _draw_histogram_panel(ax1, sbps, '收缩压 SBP 预测误差', (-50, 50),
                          ['0 %', '3.67 %', '7.34 %', '11.01 %',
                           '14.67 %', '18.34 %', '22.01 %'],
                          draw_thresholds=False, draw_xticks=False)

    plt.show()


def _classify_bp(value, bounds):
    """
    依据阈值对血压值分级。

    参数:
        value: 血压值（mmHg）
        bounds: 分级阈值二元组 (正常上限, 高血压前期上限)

    返回:
        str: 分级类别
    """
    norm_upper, pre_upper = bounds
    if value <= norm_upper:
        return '正常血压'
    if value <= pre_upper:
        return '高血压前期'
    return '高血压'


def _plot_confusion_matrix(ax, cls_gt, cls_pred, title):
    """
    绘制归一化混淆矩阵（行归一化到 0~1）。

    参数:
        ax: 绘图坐标轴
        cls_gt: 真实类别列表
        cls_pred: 预测类别列表
        title: 子图标题
    """
    cm = confusion_matrix(cls_gt, cls_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap='GnBu')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    ax.figure.colorbar(im, cax=cax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=BP_CLASSES, yticklabels=BP_CLASSES)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('真实类别', fontsize=16)
    ax.set_xlabel('预测类别', fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=15, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), fontsize=15)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > thresh else "black")
    ax.grid(False)


def evaluate_BP_Classification():
    """
    评估血压分级准确率：分别基于舒张压（DBP）与收缩压（SBP）将预测结果
    划分为正常血压 / 高血压前期 / 高血压三类，输出分类报告与混淆矩阵。
    """
    _, Y_test, _, _, max_abp, min_abp, _, Y_pred = _load_test_data()

    # --- 基于 DBP 的分级 ---
    cls_gt = []
    cls_pred = []

    for i in range(len(Y_test)):
        y_t = Y_test[i].ravel()
        y_p = Y_pred[i].ravel()

        dbp_gt = max_abp * min(y_t) + min_abp
        dbp_pred = max_abp * min(y_p) + min_abp

        cls_gt.append(_classify_bp(dbp_gt, DBP_CLASS_BOUNDS))
        cls_pred.append(_classify_bp(dbp_pred, DBP_CLASS_BOUNDS))

    print('舒张压 DBP 分级准确率')
    print(classification_report(cls_gt, cls_pred, digits=5))

    fig = plt.figure(figsize=(16, 6), dpi=120)
    ax = plt.subplot(1, 2, 1)
    _plot_confusion_matrix(ax, cls_gt, cls_pred,
                           '基于 DBP 的高血压分级准确率')

    # --- 基于 SBP 的分级 ---
    cls_gt = []
    cls_pred = []

    for i in range(len(Y_test)):
        y_t = Y_test[i].ravel()
        y_p = Y_pred[i].ravel()

        sbp_gt = max_abp * max(y_t) + min_abp
        sbp_pred = max_abp * max(y_p) + min_abp

        cls_gt.append(_classify_bp(sbp_gt, SBP_CLASS_BOUNDS))
        cls_pred.append(_classify_bp(sbp_pred, SBP_CLASS_BOUNDS))

    print('收缩压 SBP 分级准确率')
    print(classification_report(cls_gt, cls_pred, digits=5))

    ax = plt.subplot(1, 2, 2)
    _plot_confusion_matrix(ax, cls_gt, cls_pred,
                           '基于 SBP 的高血压分级准确率')

    fig.tight_layout()
    plt.show()


def _bland_altman(data1, data2):
    """
    计算并绘制 Bland-Altman 一致性图。

    参数:
        data1: 目标值序列
        data2: 估计值序列
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                    # 差值
    md = np.mean(diff)                      # 差值均值
    sd = np.std(diff, axis=0)               # 差值标准差

    plt.scatter(mean, diff, alpha=0.1, s=4)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.ylim(ymin=-75, ymax=75)
    plt.xlabel('目标值与估计值的均值 (mmHg)', fontsize=14)
    plt.ylabel('预测误差 (mmHg)', fontsize=14)
    print('一致性界限（均值+1.96SD，均值-1.96SD）：', md + 1.96 * sd, md - 1.96 * sd)


def bland_altman_plot():
    """
    绘制 SBP / DBP / MAP 预测的 Bland-Altman 一致性分析图。
    """
    _, Y_test, _, _, max_abp, min_abp, _, Y_pred = _load_test_data()

    Y_true = Y_test * max_abp + min_abp
    Y_pred = Y_pred * max_abp + min_abp

    sbpTrues = []
    sbpPreds = []
    dbpTrues = []
    dbpPreds = []
    mapTrues = []
    mapPreds = []

    for i in range(len(Y_true)):
        y_t = Y_true[i].ravel()
        y_p = Y_pred[i].ravel()

        sbpTrues.append(max(y_t))
        sbpPreds.append(max(y_p))
        dbpTrues.append(min(y_t))
        dbpPreds.append(min(y_p))
        mapTrues.append(np.mean(y_t))
        mapPreds.append(np.mean(y_p))

    fig = plt.figure(figsize=(18, 5), dpi=120)
    plt.subplot(1, 3, 1)
    print('--------- 舒张压 DBP ---------')
    _bland_altman(dbpTrues, dbpPreds)
    plt.title('舒张压 DBP 预测的 Bland-Altman 图', fontsize=18)

    plt.subplot(1, 3, 2)
    print('--------- 平均动脉压 MAP ---------')
    _bland_altman(mapTrues, mapPreds)
    plt.title('平均动脉压 MAP 预测的 Bland-Altman 图', fontsize=18)

    plt.subplot(1, 3, 3)
    print('--------- 收缩压 SBP ---------')
    _bland_altman(sbpTrues, sbpPreds)
    plt.title('收缩压 SBP 预测的 Bland-Altman 图', fontsize=18)

    plt.show()


def regression_plot():
    """
    绘制 SBP / DBP / MAP 目标值与估计值的回归图，并输出
    线性回归统计量（r 值、p 值等）。
    """
    _, Y_test, _, _, max_abp, min_abp, _, Y_pred = _load_test_data()

    Y_true = Y_test * max_abp + min_abp
    Y_pred = Y_pred * max_abp + min_abp

    sbpTrues = []
    sbpPreds = []
    dbpTrues = []
    dbpPreds = []
    mapTrues = []
    mapPreds = []

    for i in range(len(Y_true)):
        y_t = Y_true[i].ravel()
        y_p = Y_pred[i].ravel()

        sbpTrues.append(max(y_t))
        sbpPreds.append(max(y_p))
        dbpTrues.append(min(y_t))
        dbpPreds.append(min(y_p))
        mapTrues.append(np.mean(y_t))
        mapPreds.append(np.mean(y_p))

    # 绘制回归图
    plt.figure(figsize=(18, 6), dpi=120)

    plt.subplot(1, 3, 1)
    sns.regplot(dbpTrues, dbpPreds,
                scatter_kws={'alpha': 0.2, 's': 1}, line_kws={'color': '#e0b0b4'})
    plt.xlabel('目标值 (mmHg)', fontsize=14)
    plt.ylabel('估计值 (mmHg)', fontsize=14)
    plt.title('舒张压 DBP 预测回归图', fontsize=18)

    plt.subplot(1, 3, 2)
    sns.regplot(mapTrues, mapPreds,
                scatter_kws={'alpha': 0.2, 's': 1}, line_kws={'color': '#e0b0b4'})
    plt.xlabel('目标值 (mmHg)', fontsize=14)
    plt.ylabel('估计值 (mmHg)', fontsize=14)
    plt.title('平均动脉压 MAP 预测回归图', fontsize=18)

    plt.subplot(1, 3, 3)
    sns.regplot(sbpTrues, sbpPreds,
                scatter_kws={'alpha': 0.2, 's': 1}, line_kws={'color': '#e0b0b4'})
    plt.xlabel('目标值 (mmHg)', fontsize=14)
    plt.ylabel('估计值 (mmHg)', fontsize=14)
    plt.title('收缩压 SBP 预测回归图', fontsize=18)
    plt.show()

    # 输出线性回归统计量（r 值、p 值等）
    print('舒张压 DBP')
    print(scipy.stats.linregress(dbpTrues, dbpPreds))
    print('平均动脉压 MAP')
    print(scipy.stats.linregress(mapTrues, mapPreds))
    print('收缩压 SBP')
    print(scipy.stats.linregress(sbpTrues, sbpPreds))


def main():
    predicting_ABP_waveform()       # 交互式可视化预测波形
    evaluate_BHS_Standard()         # BHS 标准评估
    evaluate_AAMI_Standard()        # AAMI 标准评估
    evaluate_BP_Classification()    # 血压分级准确率评估
    bland_altman_plot()             # Bland-Altman 一致性图
    regression_plot()               # 回归图


if __name__ == '__main__':
    main()
