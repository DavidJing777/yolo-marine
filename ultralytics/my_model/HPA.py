import torch
from torch import nn


# 论文题目：A Synergistic CNN-Transformer Network with Pooling Attention Fusion for Hyperspectral Image Classification
# 论文链接：https://ieeexplore.ieee.org/document/10651326
# 官方github：https://github.com/chenpeng052/synergisticNet/blob/main/model.py
# 代码改进者：一勺汤

class HPA(nn.Module):
    def __init__(self, channels, c2=None, factor=16):
        """
        初始化 HPA 模块。

        参数:
        channels (int): 输入特征图的通道数。
        c2 (int, 可选): 未使用的参数，默认值为 None。
        factor (int, 可选): 分组因子，默认值为 32。
        """
        super(HPA, self).__init__()
        # 分组数量
        self.groups = factor
        # 确保通道数能被分组数整除
        assert channels // self.groups > 0
        # 用于计算注意力权重的 softmax 函数
        self.softmax = nn.Softmax(-1)
        # 自适应平均池化，将特征图池化为 1x1 大小
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 自适应最大池化，将特征图池化为 1x1 大小
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        # 沿高度方向进行自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Y avg
        # 沿宽度方向进行自适应平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # X avg
        # 沿高度方向进行自适应最大池化
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  # Y avg
        # 沿宽度方向进行自适应最大池化
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  # X avg

        # 组归一化层
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1 卷积层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入特征图，形状为 (b, c, h, w)。

        返回:
        torch.Tensor: 输出特征图，形状与输入相同 (b, c, h, w)。
        """
        # 获取输入特征图的批量大小、通道数、高度和宽度
        b, c, h, w = x.size()
        # 将输入特征图进行分组
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w  --->2048,2,11,11
        # 沿高度方向进行平均池化
        x_h = self.pool_h(group_x)  # 2048,2,11,1
        # 沿宽度方向进行平均池化并交换维度
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 2048,2,1,11--->2048,2,11,1
        # 将 x_h 和 x_w 拼接后通过 1x1 卷积
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 2048,2,22,1
        # 将拼接后的特征图拆分为 x_h 和 x_w
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 2048,2,11,1
        # 计算注意力并应用组归一化
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 2048,2,11,11
        # 通过 3x3 卷积
        x2 = self.conv3x3(group_x)  # 2048,2,11,11

        # 沿高度方向进行最大池化
        y_h = self.max_h(group_x)  # 2048,2,11,1
        # 沿宽度方向进行最大池化并交换维度
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        # 将 y_h 和 y_w 拼接后通过 1x1 卷积
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))  # 2048,2,22,1
        # 将拼接后的特征图拆分为 y_h 和 y_w
        y_h, y_w = torch.split(yhw, [h, w], dim=2)  # 2048,2,11,1
        # 计算注意力并应用组归一化
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid())  # 2048,2,11,11
        # 调整形状为 (b*g, c//g, hw)
        y11 = y1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        # 计算最大池化后的注意力权重
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,1,2

        # 调整形状为 (b*g, c//g, hw)
        x11 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        # 计算平均池化后的注意力权重
        x12 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,2,1,1-->2048,2,1--->2048,1,2
        # 调整形状为 (b*g, c//g, hw)
        x21 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  #2048,2,121
        # 计算平均池化后的注意力权重
        x22 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,2,1,1-->2048,2,1--->2048,1,2
        # 计算注意力权重
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        # 应用注意力权重并调整形状为输入相同的形状
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


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

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_HPA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = HPA(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_HPA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_HPA(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 定义输入特征图的通道数
    channels = 64
    # 创建 HPA 模块实例
    hpa = HPA(channels)
    # 生成随机输入特征图
    input_tensor = torch.randn(4, 64, 27, 16)
    # 进行前向传播
    output = hpa(input_tensor)
    # 打印输入和输出的形状
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")