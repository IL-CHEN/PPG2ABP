"""
模型定义模块
============

定义了论文实验中使用的全部网络结构：

- U-Net 系列（1D）：``UNet``、``UNetLite``（浅层）、``UNetWide40``、
  ``UNetWide48``、``UNetWide64``（加宽）、``UNetDS64``（加宽 + 深度监督）
- MultiResUNet 系列（1D）：``MultiResUNet1D``、``MultiResUNetDS``（深度监督）

约定
----
- 所有模型签名统一为 ``(length, n_channel=1)``，``length`` 为输入信号长度。
- 深度监督模型的输出为五路：``[out, level1, level2, level3, level4]``，
  分别对应全分辨率与逐级 2 倍下采样（长度 1/2、1/4、1/8、1/16）的预测。
- **结构兼容性声明**：为保证已训练好的 ``.h5`` 权重文件可以原样加载
  （Keras ``load_weights`` 按层名匹配），各模型的张量计算图与既定设计
  保持严格一致，本次工程化仅做代码整理与注释中文化，未改动任何层的
  连接顺序、卷积核尺寸与激活方式。
"""

from keras.layers import (Input, Conv1D, MaxPooling1D, UpSampling1D,
                          concatenate, BatchNormalization, Activation, add)
from keras.models import Model


# ---------------------------------------------------------------------------
# 公共构件
# ---------------------------------------------------------------------------
def _conv_bn(inp, filters):
    """
    标准 U-Net 基本块：Conv1D(3) + BN + Conv1D(3) + BN（激活均为 ReLU）。
    与既定实现逐层等价，保证自动生成的层名（conv1d / batch_normalization 系列）
    与既定结构完全一致，从而兼容已保存的权重文件。
    """
    out = Conv1D(filters, 3, activation='relu', padding='same')(inp)
    out = BatchNormalization()(out)
    out = Conv1D(filters, 3, activation='relu', padding='same')(out)
    out = BatchNormalization()(out)
    return out


def _unet_body(inputs, base, deep_supervision=False):
    """
    构建 5 层 U-Net 的编码器—解码器主体（UnetLite 之外的 U-Net 变体共用）。

    参数:
        inputs: 输入层
        base: 基础卷积核数量（第一层滤波器数）
        deep_supervision: 是否启用深度监督输出（多输出头）

    返回:
        单输出时返回末尾 1x1 卷积输出层；深度监督时返回 (out, level1, level2, level3, level4)
    """
    x = base

    # --- 编码器（下采样路径）---
    conv1 = _conv_bn(inputs, x)                      # 第 1 层，长度 L
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = _conv_bn(pool1, x * 2)                   # 第 2 层，长度 L/2
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = _conv_bn(pool2, x * 4)                   # 第 3 层，长度 L/4
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = _conv_bn(pool3, x * 8)                   # 第 4 层，长度 L/8
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = _conv_bn(pool4, x * 16)                  # 瓶颈层，长度 L/16

    # 深度监督：瓶颈层输出 level4（长度 L/16）
    level4 = Conv1D(1, 1, name="level4")(conv5) if deep_supervision else None

    # --- 解码器（上采样路径）---
    up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
    conv6 = _conv_bn(up6, x * 8)                     # 长度 L/8
    level3 = Conv1D(1, 1, name="level3")(conv6) if deep_supervision else None

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
    conv7 = _conv_bn(up7, x * 4)                     # 长度 L/4
    level2 = Conv1D(1, 1, name="level2")(conv7) if deep_supervision else None

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = _conv_bn(up8, x * 2)                     # 长度 L/2
    level1 = Conv1D(1, 1, name="level1")(conv8) if deep_supervision else None

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = _conv_bn(up9, x)                         # 长度 L

    out = Conv1D(1, 1, name="out" if deep_supervision else None)(conv9)

    if deep_supervision:
        return out, level1, level2, level3, level4
    return out


def conv2d_bn(x, filters, padding='same', activation='relu', name=None):
    """
    MultiResUNet 内部使用的卷积 + BN 块（1D，卷积核固定为 3）。

    参数:
        x: 输入张量
        filters: 滤波器数量
        padding: 填充方式
        activation: 激活函数；为 None 时仅返回 BN 输出（用于残差捷径）
        name: 激活层名称

    返回:
        卷积处理后的张量
    """
    kernel = 3
    x = Conv1D(filters, kernel, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation is None:
        return x
    return Activation(activation, name=name)(x)


def MultiResBlock(U, inp, alpha=2.5):
    """
    MultiRes 块：多尺度（3/5/7 卷积感受野）特征提取 + 残差捷径连接。

    参数:
        U: 对应 U-Net 同层级的滤波器数量
        inp: 输入张量

    返回:
        处理后的输出张量
    """
    W = alpha * U

    # 残差捷径：1x1 卷积对齐通道数
    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5),
                         activation=None, padding='same')

    # 多尺度卷积级联（论文中命名为 3x3 / 5x5 / 7x7，1D 实现中均为核 3 的连续卷积）
    conv3x3 = conv2d_bn(inp, int(W * 0.167), activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = BatchNormalization()(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)
    return out


def ResPath(filters, length, inp):
    """
    ResPath：由若干残差单元组成的路径，用于连接编码器与解码器的同层级特征。

    参数:
        filters: 滤波器数量
        length: 残差单元个数
        inp: 输入张量

    返回:
        处理后的输出张量
    """
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, activation=None, padding='same')
    out = conv2d_bn(inp, filters, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    for _ in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, activation=None, padding='same')
        out = conv2d_bn(out, filters, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)
    return out


def _multi_res_unet_body(inputs, deep_supervision=False):
    """
    构建 5 层 MultiResUNet 主体（MultiResUNet1D / MultiResUNetDS 共用）。

    参数:
        inputs: 输入层
        deep_supervision: 是否启用深度监督输出

    返回:
        单输出时返回末尾 1x1 卷积输出层；深度监督时返回 (out, level1, ..., level4)
    """
    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    # 深度监督：各层级输出（长度依次为 L/16, L/8, L/4, L/2）
    level4 = Conv1D(1, 1, name="level4")(mresblock5) if deep_supervision else None

    up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
    mresblock6 = MultiResBlock(32 * 8, up6)
    level3 = Conv1D(1, 1, name="level3")(mresblock6) if deep_supervision else None

    up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
    mresblock7 = MultiResBlock(32 * 4, up7)
    level2 = Conv1D(1, 1, name="level2")(mresblock7) if deep_supervision else None

    up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
    mresblock8 = MultiResBlock(32 * 2, up8)
    level1 = Conv1D(1, 1, name="level1")(mresblock8) if deep_supervision else None

    up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
    mresblock9 = MultiResBlock(32, up9)

    out = Conv1D(1, 1, name="out" if deep_supervision else None)(mresblock9)

    if deep_supervision:
        return out, level1, level2, level3, level4
    return out


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
def UNet(length, n_channel=1):
    """
    标准 U-Net（1D，基础通道数 32）。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))
    out = _unet_body(inputs, base=32)
    return Model(inputs=[inputs], outputs=[out])


def UNetWide64(length, n_channel=1):
    """
    加宽 U-Net：基础通道数 64。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))
    out = _unet_body(inputs, base=64)
    return Model(inputs=[inputs], outputs=[out])


def UNetDS64(length, n_channel=1):
    """
    深度监督 U-Net：加宽（基础通道数 64）+ 五路输出（out 及 level1~level4）。
    论文中的「近似网络」（Approximation Network）。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型，输出 [out, level1, level2, level3, level4]
    """
    inputs = Input((length, n_channel))
    out, level1, level2, level3, level4 = _unet_body(inputs, base=64, deep_supervision=True)
    return Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])


def UNetWide40(length, n_channel=1):
    """
    加宽 U-Net：基础通道数 40。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))
    out = _unet_body(inputs, base=40)
    return Model(inputs=[inputs], outputs=[out])


def UNetWide48(length, n_channel=1):
    """
    加宽 U-Net：基础通道数 48。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))
    out = _unet_body(inputs, base=48)
    return Model(inputs=[inputs], outputs=[out])


def UNetLite(length, n_channel=1):
    """
    浅层 U-Net：仅 3 层下采样，参数量更小、训练更快。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))

    conv1 = _conv_bn(inputs, 32)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = _conv_bn(pool1, 64)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = _conv_bn(pool2, 128)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv5 = _conv_bn(pool3, 256)                     # 瓶颈层

    up7 = concatenate([UpSampling1D(size=2)(conv5), conv3], axis=2)
    conv7 = _conv_bn(up7, 128)

    up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
    conv8 = _conv_bn(up8, 64)

    up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
    conv9 = _conv_bn(up9, 32)

    out = Conv1D(1, 1)(conv9)
    return Model(inputs=[inputs], outputs=[out])


def MultiResUNet1D(length, n_channel=1):
    """
    1D MultiResUNet：论文中的「精炼网络」（Refinement Network）。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型
    """
    inputs = Input((length, n_channel))
    out = _multi_res_unet_body(inputs, deep_supervision=False)
    return Model(inputs=[inputs], outputs=[out])


def MultiResUNetDS(length, n_channel=1):
    """
    1D 深度监督 MultiResUNet：五路输出（out 及 level1~level4）。

    参数:
        length: 输入信号长度
        n_channel: 输入通道数，默认 1

    返回:
        keras.model: 构建好的模型，输出 [out, level1, level2, level3, level4]
    """
    inputs = Input((length, n_channel))
    out, level1, level2, level3, level4 = _multi_res_unet_body(inputs, deep_supervision=True)
    return Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])


# ---------------------------------------------------------------------------
# 模型注册表：供训练脚本按名称索引
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    'UNet': UNet,
    'UNetLite': UNetLite,
    'UNetWide40': UNetWide40,
    'UNetWide48': UNetWide48,
    'UNetDS64': UNetDS64,
    'UNetWide64': UNetWide64,
    'MultiResUNet1D': MultiResUNet1D,
    'MultiResUNetDS': MultiResUNetDS,
}
