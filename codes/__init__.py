"""
PPG2ABP 代码包
==============

基于全卷积神经网络（U-Net / MultiResUNet）的 PPG 信号到 ABP 血压波形
翻译方法，参考论文：

    PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial
    Blood Pressure (ABP) Waveforms using Fully Convolutional Neural
    Networks (arXiv:2005.01669)

模块说明：
    config              全局路径与超参数配置
    models              模型定义（U-Net 系列 / MultiResUNet 系列）
    metrics             评估指标（BHS 标准）
    helper_functions    数据与标签预处理辅助函数
    data_processing     原始数据清洗、降采样、信号片段提取与合并
    data_handling       10 折交叉验证数据切分
    train_models        近似网络 / 精炼网络训练（10 折交叉验证）
    predict_test        测试集预测
    evaluate            多种指标评估与可视化
"""

__version__ = "1.0.0"
__author__ = ("IL-CHEN",)
