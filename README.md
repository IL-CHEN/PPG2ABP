# PPG2ABP

> **Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks**

**作者：IL-CHEN**

本项目是一个完整的 PPG → ABP 血压波形预测系统，基于全卷积神经网络（U-Net / MultiResUNet 系列）实现，采用 **Keras（TensorFlow 后端）**。本仓库由作者本人设计、开发并完整构建，包含数据处理、模型训练、测试评估的全流程代码、中文说明文档与可发布的工程化结构。

## 项目简介

心血管疾病是全球最主要的死亡原因之一。连续血压监测是最可行的方案，但传统无创技术精度不足、有创方式又复杂且存在可靠性隐患。本项目提出 **PPG2ABP**：一种基于全卷积神经网络的两阶段级联方法，可以从无创的光电容积脉搏波（PPG）信号估计连续的动脉血压（ABP）波形，**平均绝对误差为 4.604 mmHg**，且能同时保持波形的形态、幅度与相位。

更突出的成果是：由估计波形计算出的舒张压（DBP）、平均动脉压（MAP）与收缩压（SBP）在多种指标下均优于既有工作（平均绝对误差分别为 3.449 ± 6.147、2.310 ± 4.437、5.727 ± 9.162 mmHg），尽管 PPG2ABP 并未显式针对这些数值进行训练。其中 **DBP 与 MAP 均达到 BHS（英国高血压学会）标准的 A 级**，并满足 **AAMI（美国医疗器械促进协会）标准**。

## 主要特性

- **完整流水线**：覆盖原始数据处理 → 10 折交叉验证切分 → 两阶段级联训练（近似网络 + 精炼网络）→ 测试集预测 → 多指标评估的全流程；
- **全量中文化**：全部代码注释、文档与用户界面文本均为中文；
- **工程化架构**：配置集中管理（`config.py`）、显式导入、模块化包结构、去除冗余代码；
- **可发布结构**：规范的 README、LICENSE、`.gitignore`、`requirements.txt`，数据与模型大文件不入库。

## 目录结构

```
PPG2ABP/
├── .gitignore              # Git 忽略规则（数据/模型/IDE 等不入库）
├── LICENSE                 # MIT 许可证
├── README.md               # 本文件
├── requirements.txt        # 依赖清单
├── codes/                  # 全部代码与 Notebook
│   ├── config.py           # 全局路径与超参数集中配置
│   ├── data_processing.py  # 数据处理：清洗、降采样、片段提取与合并
│   ├── data_handling.py    # 10 折交叉验证数据切分
│   ├── models.py           # 模型定义（U-Net / MultiResUNet 系列）
│   ├── metrics.py          # BHS 标准指标
│   ├── helper_functions.py # 两阶段训练的数据与标签预处理
│   ├── train_models.py     # 近似网络 / 精炼网络训练
│   ├── predict_test.py     # 测试集预测
│   ├── evaluate.py         # 多种指标评估与可视化
│   └── PPG2ABP.ipynb       # 完整流水线演示 Notebook
└── data/                   # 【不入库】数据与模型（见下方数据准备）
```

> 注：数据与模型文件体积巨大（`data.hdf5` 约 2.5 GB，模型权重 100+ MB），已通过 `.gitignore` 排除，不会进入 Git 历史。

## 环境要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.5 / 3.6 | TensorFlow 1.5 不支持更高版本 Python |
| TensorFlow | 1.5.0（GPU） | 或 `tensorflow==1.5.0`（CPU） |
| Keras | 2.2.4 | 深度监督多输出等接口依赖该版本行为 |
| CUDA / cuDNN | 与 TF 1.5 配套 | GPU 训练必需 |

**重要**：本项目代码基于 TensorFlow 1.x / Keras 2.2.4 时代的 API 实现，请勿升级深度学习框架，否则可能出现接口不兼容。

### 安装

```bash
# 建议新建 Python 3.6 虚拟环境后执行
pip install -r requirements.txt
```

## 数据准备

### 方式一：全流程（从原始数据开始）

1. 下载 UCI 数据集：<https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip>
2. 解压到 `raw_data/` 目录（含 `Part_1.mat` ~ `Part_4.mat`）。
3. 按 `codes/PPG2ABP.ipynb` 第 2 节依次执行数据处理（`process_data` → `downsample_data` → `extract_episodes` → `merge_episodes`）。

> ⚠️ 数据处理计算量极大（原实验环境约需 9~11 天），如非必要，建议直接采用方式二。

### 方式二：使用已编译好的精简数据（推荐）

| 文件 | 用途 | 下载链接 |
| --- | --- | --- |
| `data/data.hdf5` | 处理完成的信号数据集（约 2.5 GB） | [Google Drive](https://drive.google.com/file/d/1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X/view?usp=sharing) |
| `data/test.p` | 独立测试集（约 446 MB） | [Google Drive](https://drive.google.com/file/d/1YC9Qv7AWo20DULCEA4OxTCzxkGc-u7OY/view?usp=sharing) |
| `data/meta9.p` | 第 10 折元数据（信号极值） | [Google Drive](https://drive.google.com/file/d/1bO93Z6r1rqZUZC3zQdvHQSSCjap_muia/view?usp=sharing) |
| `models/ApproximateNetwork.h5` | 最优近似网络权重 | [Google Drive](https://drive.google.com/file/d/1R0t3VxPLBpQmIKKyH9ulDdr9irwsyOpD/view?usp=sharing) |
| `models/RefinementNetwork.h5` | 最优精炼网络权重 | [Google Drive](https://drive.google.com/file/d/1qc97paeXHDWrOEsPR1_2tl7WtMcJxQbF/view?usp=sharing) |

（如需 `processed_data.zip`、`candidates.p` 等中间产物，见 Notebook 内各阶段链接。）

## 快速开始

以下操作均在 `codes/` 目录下执行（脚本路径基于当前工作目录解析，见 `config.py`）。

### 场景一：仅评估（最快，推荐体验）

```bash
cd codes
# 1. 下载 data/test.p、data/meta9.p，放入 data/ 目录
# 2. 下载 ApproximateNetwork.h5、RefinementNetwork.h5，放入 models/ 目录
# 3. 生成测试集预测（避免评估时重复计算）
python predict_test.py
# 4. 交互式可视化单条样本波形
python evaluate.py        # 或进入 PPG2ABP.ipynb 按第 5~6 节执行
```

### 场景二：训练 + 评估

```bash
cd codes
# 1. 准备 data/data.hdf5（下载或自行处理），放入 data/ 目录
python data_handling.py    # 10 折交叉验证切分
python train_models.py     # 训练近似网络与精炼网络（GPU 约需 11~12 天）
# 2. 将最优折的模型对复制为 models/ApproximateNetwork.h5 与 models/RefinementNetwork.h5
python predict_test.py     # 生成测试集预测
python evaluate.py         # 评估
```

### 场景三：全流程（原始数据 → 评估）

依次执行 `codes/PPG2ABP.ipynb` 的第 1~6 节即可。

## 流水线与产物

| 阶段 | 模块 | 主要产物 |
| --- | --- | --- |
| 数据处理 | `data_processing.py` | `processed_data/*.csv`、`candidates.p`、`ppgs/*.p`、`abps/*.p`、`data/data.hdf5` |
| 数据切分 | `data_handling.py` | `data/train{n}.p`、`data/val{n}.p`、`data/meta{n}.p`、`data/test.p` |
| 模型训练 | `train_models.py` | `models/*_model1_fold{n}.h5`、`models/*_model2_fold{n}.h5`、`History/*.p` |
| 测试预测 | `predict_test.py` | `test_output_approximate.p`、`test_output.p` |
| 评估 | `evaluate.py` | BHS / AAMI 指标表、血压分级、Bland-Altman 图、回归图 |

## 评估指标

本项目在以下标准下对模型进行评估：

- **BHS 标准**（英国高血压学会）：误差 ≤5 / ≤10 / ≤15 mmHg 的样本占比；
- **AAMI 标准**（美国医疗器械促进协会）：误差的均值（ME）与标准差（STD）；
- **血压分级**：基于 DBP / SBP 的正常血压、高血压前期、高血压三级分类准确率；
- **Bland-Altman 图**与**回归图**：一致性分析与线性回归统计量（r 值、p 值）。

## 复现说明

使用 GPU 训练深度网络是**非确定性的**，在相同数据上重新训练很难保证得到与论文完全一致的模型。如需复现论文结果，建议使用项目提供的模型与数据切分（原始数据切分曾并行执行，随机过程无法固定随机种子）。

## 引用

如果您在项目或研究中使用 **PPG2ABP**，请引用以下论文：

```bibtex
@article{ibtehaz2020ppg2abp,
  title={PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={arXiv preprint arXiv:2005.01669},
  year={2020}
}
```

## 许可

本项目基于 [MIT License](LICENSE) 发布。
