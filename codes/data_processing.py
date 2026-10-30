"""
数据处理模块
============

将 UCI 数据库下载的原始数据（``Part_1.mat`` ~ ``Part_4.mat``）逐级加工为
可供训练使用的信号片段：

1. ``process_data``：切分 10 秒信号片段并计算每段的收缩压/舒张压，输出 CSV；
2. ``observe_processed_data``：可视化处理后数据的血压分布；
3. ``downsample_data``：按论文方案基于血压分箱降采样；
4. ``extract_episodes``：从原始数据中提取候选片段的 PPG / ABP 波形；
5. ``merge_episodes``：合并所有片段为 ``data.hdf5``。

注意：数据处理步骤计算量极大，
建议直接下载已编译好的 ``data.hdf5`` 以跳过 2A~2E 步骤（见根目录 README）。
"""

import h5py
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR,
                    PPG_DIR, ABP_DIR, CANDIDATES_FILE,
                    SAMPLING_FREQUENCY, EPISODE_LENGTH, EPISODE_STEP,
                    DATA_PARTS, DOWNSAMPLE_RATIO, DOWNSAMPLE_MIN_THRESH)

sns.set()


def process_data():
    """
    提取 10 秒长信号片段的收缩压（SBP）与舒张压（DBP）值，并保存为 CSV 文件。

    每个记录中，每隔 5 秒取一个 10 秒片段；每段记录为一行
    ``{起始采样点},{SBP},{DBP}``，写入 ``processed_data/Part_{k}_{i}.csv``。
    """
    # 每段信号的采样点数与相邻片段起始点的步长（采样点）
    samples_in_episode = round(SAMPLING_FREQUENCY * EPISODE_LENGTH)
    d_samples = round(SAMPLING_FREQUENCY * EPISODE_STEP)

    # 创建 processed_data 目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 依次处理 4 份原始数据文件
    for k in range(1, DATA_PARTS + 1):
        print("正在处理第 {}/{} 份数据文件".format(k, DATA_PARTS))

        f = h5py.File(os.path.join(RAW_DATA_DIR, 'Part_{}.mat'.format(k)), 'r')
        ky = 'Part_' + str(k)                       # HDF5 文件中的数据集键

        for i in tqdm(range(len(f[ky])), desc='读取记录'):
            signal = []                             # PPG 信号
            bp = []                                 # ABP 信号

            output_str = '10s,SBP,DBP\n'            # CSV 文件头

            # 读取该记录中的全部采样点
            for j in tqdm(range(len(f[f[ky][i][0]])),
                          desc='读取记录 {}/3000 中的采样点'.format(i + 1)):
                signal.append(f[f[ky][i][0]][j][0])  # PPG 信号
                bp.append(f[f[ky][i][0]][j][1])      # ABP 信号

            # 每隔 d_samples 取一个 10 秒片段，计算片段的 SBP 与 DBP
            for j in tqdm(range(0, len(f[f[ky][i][0]]) - samples_in_episode, d_samples),
                          desc='处理记录 {}/3000 中的片段'.format(i + 1)):
                sbp = max(bp[j:j + samples_in_episode])    # SBP 值
                dbp = min(bp[j:j + samples_in_episode])    # DBP 值

                output_str += '{},{},{}\n'.format(j, sbp, dbp)  # 追加到 CSV

            fp = open(os.path.join(PROCESSED_DATA_DIR,
                                   'Part_{}_{}.csv'.format(k, i)), 'w')
            fp.write(output_str)
            fp.close()


def observe_processed_data():
    """
    观察 10 秒长片段的 SBP 与 DBP 分布（直方图）。
    """
    files = next(os.walk(PROCESSED_DATA_DIR))[2]

    sbps = []
    dbps = []

    for fl in tqdm(files, desc='浏览处理结果文件'):
        lines = open(os.path.join(PROCESSED_DATA_DIR, fl), 'r').read().split('\n')[1:-1]

        for line in tqdm(lines, desc='浏览文件中的片段'):
            values = line.split(',')
            sbp = int(float(values[1]))
            dbp = int(float(values[2]))

            sbps.append(sbp)
            dbps.append(dbp)

    plt.subplot(2, 1, 1)
    plt.hist(sbps, bins=180)
    plt.title('收缩压 SBP')

    plt.subplot(2, 1, 2)
    plt.hist(dbps, bins=180)
    plt.title('舒张压 DBP')

    plt.show()


def downsample_data(minThresh=DOWNSAMPLE_MIN_THRESH, ratio=DOWNSAMPLE_RATIO):
    """
    按论文方案对数据降采样。

    先将片段按 SBP 与 DBP 分箱；对每个箱，若 25% 的样本数超过阈值则只随机
    保留阈值个样本，否则全部保留（阈值默认 2500）。DBP 分箱完成后，再对
    SBP 分箱补足其余样本，同时保证同一片段不会重复入选。最终得到约 127260
    个片段（合计约 353.5 小时信号）。

    参数:
        minThresh: 每个血压箱最多选取的片段数（默认 2500）
        ratio: 每个血压箱的采样比例（默认 0.25）

    结果:
        候选片段列表保存至 ``candidates.p``，并绘制选取后的血压分布图。
    """
    files = next(os.walk(PROCESSED_DATA_DIR))[2]      # 加载全部 CSV 文件

    sbps_dict = {}                                    # 各 SBP 值对应的片段列表
    dbps_dict = {}                                    # 各 DBP 值对应的片段列表
    sbps_cnt = {}                                     # 各 SBP 值的片段计数
    dbps_cnt = {}                                     # 各 DBP 值的片段计数
    dbps_taken = {}                                   # 各 DBP 值已选数量
    sbps_taken = {}                                   # 各 SBP 值已选数量

    candidates = []                                   # 候选片段列表
    lut = {}                                          # 查重表（防止重复选取）

    for fl in tqdm(files, desc='浏览文件'):
        lines = open(os.path.join(PROCESSED_DATA_DIR, fl), 'r').read().split('\n')[1:-1]

        for line in tqdm(lines, desc='读取片段'):
            values = line.split(',')
            file_no = int(fl.split('_')[1])           # 文件编号
            record_no = int(fl.split('.')[0].split('_')[2])   # 记录编号
            episode_st = int(values[0])               # 片段起始采样点
            sbp = int(float(values[1]))               # 该片段的 SBP
            dbp = int(float(values[2]))               # 该片段的 DBP

            # 维护 SBP 分箱
            if sbp not in sbps_dict:
                sbps_dict[sbp] = []
                sbps_cnt[sbp] = 0
            sbps_dict[sbp].append((file_no, record_no, episode_st))
            sbps_cnt[sbp] += 1

            # 维护 DBP 分箱（额外记录 SBP，供后续查重）
            if dbp not in dbps_dict:
                dbps_dict[dbp] = []
                dbps_cnt[dbp] = 0
            dbps_dict[dbp].append((file_no, record_no, episode_st, sbp))
            dbps_cnt[dbp] += 1

    sbp_keys = list(sbps_dict)
    dbp_keys = list(dbps_dict)
    sbp_keys.sort()
    dbp_keys.sort()

    # 第一阶段：按 DBP 分箱随机选取
    for dbp in tqdm(dbp_keys, desc='DBP 分箱'):
        cnt = min(int(dbps_cnt[dbp] * ratio), minThresh)   # 该箱应选取的数量

        for _ in tqdm(range(cnt), desc='随机选取索引'):
            indix = np.random.randint(len(dbps_dict[dbp]))  # 随机索引

            # 将该片段加入候选列表
            candidates.append([dbps_dict[dbp][indix][0],
                               dbps_dict[dbp][indix][1],
                               dbps_dict[dbp][indix][2]])

            if dbp not in dbps_taken:
                dbps_taken[dbp] = 0
            dbps_taken[dbp] += 1

            # 记录该片段的 SBP 已被选取（供第二阶段查重）
            if dbps_dict[dbp][indix][3] not in sbps_taken:
                sbps_taken[dbps_dict[dbp][indix][3]] = 0
            sbps_taken[dbps_dict[dbp][indix][3]] += 1

            # 将片段写入查重表
            if dbps_dict[dbp][indix][0] not in lut:
                lut[dbps_dict[dbp][indix][0]] = {}
            if dbps_dict[dbp][indix][1] not in lut[dbps_dict[dbp][indix][0]]:
                lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]] = {}
            lut[dbps_dict[dbp][indix][0]][dbps_dict[dbp][indix][1]][dbps_dict[dbp][indix][2]] = 1

            # 移除已选片段，避免重复选取
            dbps_dict[dbp].pop(indix)

    # 第二阶段：按 SBP 分箱补足（扣除第一阶段已选数量）
    for sbp in tqdm(sbp_keys, desc='SBP 分箱'):
        if sbp not in sbps_taken:
            sbps_taken[sbp] = 0

        cnt = min(int(sbps_cnt[sbp] * ratio), minThresh) - sbps_taken[sbp]

        for _ in tqdm(range(cnt), desc='随机选取索引'):
            while len(sbps_dict[sbp]) > 0:
                try:
                    indix = np.random.randint(len(sbps_dict[sbp]))   # 随机索引
                except ValueError:
                    pass

                # 若该片段已在 DBP 阶段被选过，则跳过
                try:
                    dumi = lut[sbps_dict[sbp][indix][0]][sbps_dict[sbp][indix][1]][sbps_dict[sbp][indix][2]]
                except KeyError:
                    sbps_dict[sbp].pop(indix)
                    continue

                candidates.append([sbps_dict[sbp][indix][0],
                                   sbps_dict[sbp][indix][1],
                                   sbps_dict[sbp][indix][2]])
                sbps_taken[sbp] += 1
                sbps_dict[sbp].pop(indix)
                break

    print('共选取了 {} 个片段'.format(len(candidates)))

    pickle.dump(candidates, open(CANDIDATES_FILE, 'wb'))     # 保存候选列表

    # 绘制降采样后的血压分布
    sbp_keys = list(sbps_taken)
    dbp_keys = list(dbps_taken)
    sbp_keys.sort()
    dbp_keys.sort()

    sbps = [sbps_taken[sbp] for sbp in sbp_keys]
    dbps = [dbps_taken[dbp] for dbp in dbp_keys]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.bar(sbp_keys, sbps)
    plt.title('收缩压 SBP')

    plt.subplot(2, 1, 2)
    plt.bar(dbp_keys, dbps)
    plt.title('舒张压 DBP')

    plt.show()


def extract_episodes(candidates):
    """
    从原始数据中提取候选片段的 PPG 与 ABP 波形。

    按候选列表依次读取，将每条波形成功保存到 ``ppgs/`` 与 ``abps/`` 目录
    （文件名为候选下标，如 ``0.p``）。

    参数:
        candidates: 候选片段列表（文件编号, 记录编号, 片段起始采样点）
    """
    os.makedirs(PPG_DIR, exist_ok=True)               # PPG 信号目录
    os.makedirs(ABP_DIR, exist_ok=True)               # ABP 信号目录

    samples_in_episode = round(SAMPLING_FREQUENCY * EPISODE_LENGTH)

    for k in tqdm(range(1, DATA_PARTS + 1), desc='读取原始文件'):
        f = h5py.File(os.path.join(RAW_DATA_DIR, 'Part_{}.mat'.format(k)), 'r')
        ky = 'Part_' + str(k)                         # HDF5 数据集键

        for indix in tqdm(range(len(candidates)), desc='读取第 {}/4 份文件'.format(k)):
            if candidates[indix][0] != k:             # 该候选属于其他文件
                continue

            record_no = int(candidates[indix][1])     # 记录编号
            episode_st = int(candidates[indix][2])    # 片段起始采样点

            ppg = []                                  # PPG 信号
            abp = []                                  # ABP 信号

            for j in tqdm(range(episode_st, episode_st + samples_in_episode),
                          desc='读取片段 {}'.format(indix)):
                ppg.append(f[f[ky][record_no][0]][j][0])   # PPG 信号
                abp.append(f[f[ky][record_no][0]][j][1])   # ABP 信号

            pickle.dump(np.array(ppg), open(os.path.join(PPG_DIR, '{}.p'.format(indix)), 'wb'))
            pickle.dump(np.array(abp), open(os.path.join(ABP_DIR, '{}.p'.format(indix)), 'wb'))


def merge_episodes():
    """
    合并提取的片段为 ``data.hdf5`` 文件。

    读取 ``abps/`` 与 ``ppgs/`` 目录下的全部片段，随机打乱顺序后写入
    HDF5 数据集（每条记录为 ``[ABP, PPG]``）。此处的随机打乱保证下一步
    数据切分可以按区间直接划分训练/验证/测试集。
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    files = next(os.walk(ABP_DIR))[2]
    np.random.shuffle(files)                          # 随机打乱（便于后续直接切分）

    data = []
    for fl in tqdm(files):
        abp = pickle.load(open(os.path.join(ABP_DIR, fl), 'rb'))
        ppg = pickle.load(open(os.path.join(PPG_DIR, fl), 'rb'))
        data.append([abp, ppg])

    f = h5py.File(os.path.join(DATA_DIR, 'data.hdf5'), 'w')
    dset = f.create_dataset('data', data=data)


def main():
    process_data()
    observe_processed_data()
    downsample_data()
    candidates = pickle.load(open(CANDIDATES_FILE, 'rb'))
    extract_episodes(candidates)
    merge_episodes()


if __name__ == '__main__':
    main()
