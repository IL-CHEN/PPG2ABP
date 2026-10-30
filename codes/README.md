# PPG2ABP 代码与流水线说明

PPG2ABP 的完整流水线使用 **Python 3.5.2** 实现，深度学习模型基于 **Keras（TensorFlow 后端）** 构建。本目录下的脚本需在 `codes/` 目录内运行（路径基于当前工作目录解析，详见 `config.py`）。

## 模块一览

| 脚本 | 作用 |
| --- | --- |
| `config.py` | 全局路径与超参数集中配置（采样率、信号长度、折数、批大小等） |
| `data_processing.py` | 处理原始数据并降采样 |
| `data_handling.py` | 将数据切分为训练深度网络所需的数据集 |
| `models.py` | 包含多种网络模型定义（U-Net / MultiResUNet 系列） |
| `metrics.py` | 辅助脚本，包含指标计算（BHS 标准） |
| `helper_functions.py` | 辅助脚本，包含两阶段训练的数据与标签预处理函数 |
| `train_models.py` | 使用 10 折交叉验证训练模型 |
| `predict_test.py` | 生成测试数据的预测结果 |
| `evaluate.py` | 在多种指标下评估 PPG2ABP |

## 系统要求

开发测试环境：Intel Core i7-7700（3.6 GHz，8 MB 缓存）CPU、16 GB 内存、NVIDIA TITAN XP（12 GB，1582 MHz）GPU，Ubuntu 16.04 操作系统。

- **训练模型必须有 GPU**；
- 仅测试 / 评估时无需 GPU。

## 安装依赖

```bash
pip3 install -r requirements.txt
```

依赖版本以 `requirements.txt` 为准（TensorFlow 1.5.0 / Keras 2.2.4 时代版本，请勿升级，详见根目录 README）。

## 下载原始数据集

数据集位于 UCI 机器学习数据库：

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip
unzip data.zip -d raw_data
```

（也可用其他解压软件解压，并将文件放入 `raw_data/` 目录。）

## 使用已编译好的精简数据（推荐）

下载降采样后的数据集 [data.hdf5](https://drive.google.com/file/d/1IxN2sX2TX0uK6CFDh8eudb8haz3RlF7X/view?usp=sharing) 放入 `data/` 目录，即可跳过耗时的数据处理步骤。进一步说明请参考 [PPG2ABP.ipynb](PPG2ABP.ipynb)。

## 脚本使用步骤

下载并解压数据（放入 `raw_data/` 目录）后，依次运行以下脚本：

1. 处理数据（i7-7700 CPU 上约需 9~11 天）

```bash
python3 data_processing.py
```

2. 切分数据（10 折交叉验证）

```bash
python3 data_handling.py
```

3. 训练模型（NVIDIA TITAN XP GPU 上约需 11~12 天）

```bash
python3 train_models.py
```

4. 生成测试数据预测

```bash
python3 predict_test.py
```

5. 在多种指标下评估 PPG2ABP

```bash
python3 evaluate.py
```

## 预期结果

预期结果展示在 Jupyter Notebook [PPG2ABP.ipynb](PPG2ABP.ipynb) 的最后一部分。

## 演示

完整的演示与分步说明请参考 [PPG2ABP.ipynb](PPG2ABP.ipynb)（已中文化）。

## 注意事项

使用 GPU 训练深度网络是**非确定性的**过程，在相同数据上重新训练很难保证得到与论文完全一致的模型。为复现论文结果，建议使用项目提供的模型与数据切分（原始数据切分曾并行执行，随机过程无法固定随机种子）。更多细节请参考 Notebook。
