import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention
# 论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf
# 官方github： https://github.com/Inha-CVAI/MADGNet/blob/main/IS2D_models/mfmsnet.py
# 代码改进者：一勺汤


def get_freq_indices(method):
    """
    根据方法名称获取频率索引。

    参数:
    - method: str, 指定频率选择方法，例如 'top1', 'top2', 'low4', 'bot8' 等。

    返回:
    - mapper_x: list, 频率索引的 x 坐标。
    - mapper_y: list, 频率索引的 y 坐标。
    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])  # 提取方法名称中的频率数量
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        """
        初始化 MultiFrequencyChannelAttention 类。

        参数:
        - in_channels: int, 输入特征图的通道数。
        - dct_h: int, DCT 变换的高度。
        - dct_w: int, DCT 变换的宽度。
        - frequency_branches: int, 频率分支的数量，必须是 1, 2, 4, 8, 16, 32 之一。
        - frequency_selection: str, 频率选择方法，例如 'top', 'low', 'bot'。
        - reduction: int, 通道缩减比例。
        """
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # 初始化 DCT 滤波器
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx),
                                 self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        # 定义池化层
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: torch.Tensor, 输入特征图，形状为 (batch_size, in_channels, H, W)。

        返回:
        - torch.Tensor, 经过多频率通道注意力机制处理后的特征图。
        """
        batch_size, C, H, W = x.shape

        x_pooled = x

        # 如果输入特征图的大小与 DCT 变换的大小不匹配，则进行自适应池化
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 计算多频谱特征
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 通过全连接层生成注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        # 合并注意力图并应用 Sigmoid 激活函数
        multi_spectral_attention_map = F.sigmoid(
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        # 将注意力图应用到输入特征图上
        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        """
        生成 DCT 滤波器。

        参数:
        - tile_size_x: int, DCT 滤波器的高度。
        - tile_size_y: int, DCT 滤波器的宽度。
        - mapper_x: int, 频率索引的 x 坐标。
        - mapper_y: int, 频率索引的 y 坐标。
        - in_channels: int, 输入通道数。

        返回:
        - dct_filter: torch.Tensor, 生成的 DCT 滤波器。
        """
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        """
        构建 DCT 滤波器的基函数。

        参数:
        - pos: int, 当前位置。
        - freq: int, 频率索引。
        - POS: int, 总位置数。

        返回:
        - float, DCT 滤波器的基函数值。
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class PSABloc_MFCA(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = MultiFrequencyChannelAttention(in_channels=c, dct_h=7, dct_w=7, frequency_branches=16,
                                          frequency_selection='top')
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x



class C2PSA_MFCA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABloc_MFCA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


def main():
    # 创建一个随机输入张量 (batch_size=1, channels=64, height=32, width=32)
    x = torch.randn(1, 64, 27, 32)

    # 初始化 MultiFrequencyChannelAttention
    mfca = MultiFrequencyChannelAttention(in_channels=64, dct_h=7, dct_w=7, frequency_branches=16,
                                          frequency_selection='top')

    # 应用多频率通道注意力机制
    output = mfca(x)

    # 打印结果
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output min value:", output.min().item())
    print("Output max value:", output.max().item())


if __name__ == "__main__":
    main()