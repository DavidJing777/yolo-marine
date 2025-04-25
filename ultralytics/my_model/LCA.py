import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# 论文题目： You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement
# 代码改进者：一勺汤

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # 可学习的权重参数，形状为normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏置参数，形状为normalized_shape
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 用于防止除零操作的小常数
        self.eps = eps
        # 数据格式，支持"channels_last"或"channels_first"
        self.data_format = data_format
        # 检查数据格式是否支持
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 归一化的形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式是channels_last，即(batch_size, height, width, channels)
        if self.data_format == "channels_last":
            # 使用PyTorch内置的LayerNorm函数进行归一化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式是channels_first，即(batch_size, channels, height, width)
        elif self.data_format == "channels_first":
            # 计算每个通道的均值，保持维度
            u = x.mean(1, keepdim=True)
            # 计算每个通道的方差，保持维度
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 归一化操作
            x = (x - u) / torch.sqrt(s + self.eps)
            # 应用权重和偏置
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 可学习的温度参数，用于调整注意力分数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 生成查询（query）的卷积层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 对查询进行深度可分离卷积
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 生成键（key）和值（value）的卷积层
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 对键和值进行深度可分离卷积
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # 生成查询
        q = self.q_dwconv(self.q(x))
        # 生成键和值
        kv = self.kv_dwconv(self.kv(y))
        # 分离键和值
        k, v = kv.chunk(2, dim=1)

        # 调整查询的形状，以便多头注意力计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 调整键的形状，以便多头注意力计算
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 调整值的形状，以便多头注意力计算
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        # 对键进行归一化
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力分数进行Softmax操作
        attn = nn.functional.softmax(attn, dim=-1)

        # 计算注意力加权的值
        out = (attn @ v)

        # 恢复输出的形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        # 隐藏层的特征数量，根据扩展因子计算
        hidden_features = int(dim * ffn_expansion_factor)

        # 输入投影层，将输入映射到隐藏层维度的2倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 对投影后的特征进行深度可分离卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 第一个分支的深度可分离卷积
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        # 第二个分支的深度可分离卷积
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        # 输出投影层，将特征映射回原始维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # Tanh激活函数
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # 输入投影
        x = self.project_in(x)
        # 对投影后的特征进行深度可分离卷积并拆分为两个分支
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 第一个分支进行Tanh激活和残差连接
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        # 第二个分支进行Tanh激活和残差连接
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        # 两个分支元素相乘
        x = x1 * x2
        # 输出投影
        x = self.project_out(x)
        return x


# 自动填充函数：确保卷积操作后输出的特征图尺寸与输入一致
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 标准卷积模块：包括卷积、批量归一化、激活函数等
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 定义卷积层，自动计算填充
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 定义批量归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 默认激活函数（SiLU）或用户自定义的激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # 依次进行卷积、批量归一化和激活操作
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        # 仅进行卷积和激活操作，跳过批量归一化
        return self.act(self.conv(x))



class LCA(nn.Module):
    def __init__(self, dim, out ,num_heads, bias=False):
        super(LCA, self).__init__()
        # 层归一化
        self.norm = LayerNorm(dim)
        # 强度增强层
        self.gdfn = IEL(dim)
        # 交叉注意力模块
        self.ffn = CAB(dim, num_heads, bias=bias)
        # 卷积上采样层，将输入通道数为dim*0.5的特征图上采样到dim通道
        self.conv_up = Conv(int(dim * 0.5), dim, 1, act=nn.ReLU())

        self.conv_down = Conv(int(dim), out, 1, act=nn.ReLU())

    def forward(self, x):
        x1, x2 = x[0], x[1]
        channel_x1 = x1.shape[1]
        channel_x2 = x2.shape[1]

        # 判断通道数是否一致，如果不一致则对通道数少的进行上采样
        if channel_x1 != channel_x2:
            if channel_x1 < channel_x2:
                x1 = self.conv_up(x1)
            else:
                x2 = self.conv_up(x2)

        # 残差连接和交叉注意力计算
        x1 = x1 + self.ffn(self.norm(x1), self.norm(x2))
        # 经过强度增强层和残差连接
        x1 = x1 + self.gdfn(self.norm(x1))

        x1 = self.conv_down(x1)
        return x1


def main():
    # # 设置随机种子以保证可重复性
    # torch.manual_seed(42)
    #
    # # 测试LayerNorm
    # print("Testing LayerNorm...")
    # norm = LayerNorm(64)
    # x = torch.randn(2, 64, 32, 32)
    # out = norm(x)
    # print(f"LayerNorm output shape: {out.shape}\n")
    #
    # # 测试CAB（Cross Attention Block）
    # print("Testing CAB...")
    # cab = CAB(dim=64, num_heads=4, bias=True)
    # x = torch.randn(2, 64, 32, 32)
    # y = torch.randn(2, 64, 32, 32)
    # out = cab(x, y)
    # print(f"CAB output shape: {out.shape}\n")
    #
    # # 测试IEL（Intensity Enhancement Layer）
    # print("Testing IEL...")
    # iel = IEL(dim=64)
    # x = torch.randn(2, 64, 32, 32)
    # out = iel(x)
    # print(f"IEL output shape: {out.shape}\n")
    #
    # # 测试HV_LCA（Horizontal-Vertical Lightweight Cross Attention）
    # print("Testing HV_LCA...")
    # hv_lca = HV_LCA(dim=64, num_heads=4)
    # x = torch.randn(2, 64, 32, 32)
    # y = torch.randn(2, 64, 32, 32)
    # out = hv_lca(x, y)
    # print(f"HV_LCA output shape: {out.shape}\n")

    # 测试I_LCA（Intensity Lightweight Cross Attention）
    print("Testing I_LCA...")
    i_lca = LCA(dim=256, num_heads=4)
    x1 = torch.randn((32, 128, 8, 8))  # 批次大小32，通道数512，特征图大小8x8
    x2 = torch.randn((32, 256, 8, 8))  # 批次大小32，通道数256，特征图大小8x8

    out = i_lca((x1, x2))
    print(f"I_LCA output shape: {out.shape}\n")


if __name__ == "__main__":
    main()
