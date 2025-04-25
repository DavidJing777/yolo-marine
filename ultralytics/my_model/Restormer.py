import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange

# https://github.com/slrl123/BSAFusion/blob/main/modal_2d/Restormer.py
# https://arxiv.org/pdf/2412.08050

# 多尺度动态注意力模块（MDTA）
class MDTA(nn.Module):
    def __init__(self, out_c):
        super(MDTA, self).__init__()
        # 第一个卷积块，用于生成查询（query）
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),  # 1x1卷积
            nn.Conv2d(out_c, out_c, 3, 1, 1)   # 3x3卷积
        )
        # 第二个卷积块，用于生成键（key）
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),  # 1x1卷积
            nn.Conv2d(out_c, out_c, 3, 1, 1)   # 3x3卷积
        )
        # 第三个卷积块，用于生成值（value）
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),  # 1x1卷积
            nn.Conv2d(out_c, out_c, 3, 1, 1)   # 3x3卷积
        )
        # 最后一个1x1卷积，用于调整输出特征图的通道数
        self.conv4 = nn.Conv2d(out_c, out_c, 1, 1, 0)

    def forward(self, x):
        x_o = x  # 保存输入，用于残差连接
        x = F.layer_norm(x, x.shape[-2:])  # 对输入进行层归一化
        C, W, H = x.size()[1], x.size()[2], x.size()[3]  # 获取通道数、宽度和高度

        # 生成查询
        q = self.conv1(x)
        q = rearrange(q, 'b c w h -> b (w h) c')  # 调整维度

        # 生成键
        k = self.conv2(x)
        k = rearrange(k, 'b c w h -> b c (w h)')  # 调整维度

        # 生成值
        v = self.conv3(x)
        v = rearrange(v, 'b c w h -> b (w h) c')  # 调整维度

        # 计算注意力矩阵
        A = torch.matmul(k, q)
        A = rearrange(A, 'b c1 c2 -> b (c1 c2)', c1=C, c2=C)
        A = torch.softmax(A, dim=1)  # 对注意力矩阵进行softmax操作
        A = rearrange(A, 'b (c1 c2) -> b c1 c2', c1=C, c2=C)

        # 计算加权值
        v = torch.matmul(v, A)
        # v = rearrange(v, 'b (h w) c -> b c h w', c=C, h=H, w=W)  # 调整维度
        v = rearrange(v, 'b (h w) c -> b c w h', c=C, w=W, h=H)  # 调整维度

        # 残差连接
        return self.conv4(v) + x_o

# 门控深度前馈网络（GDFN）
class GDFN(nn.Module):
    def __init__(self, out_c):
        super(GDFN, self).__init__()
        # 第一个卷积块，用于生成第一个特征图
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),  # 1x1卷积，通道数扩展4倍
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)  # 3x3卷积
        )
        # 第二个卷积块，用于生成第二个特征图
        self.Dconv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),  # 1x1卷积，通道数扩展4倍
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)  # 3x3卷积
        )
        # 最后一个1x1卷积，用于将通道数恢复到原始大小
        self.conv = nn.Conv2d(out_c * 4, out_c, 1, 1, 0)

    def forward(self, x):
        x_o = x  # 保存输入，用于残差连接
        x = F.layer_norm(x, x.shape[-2:])  # 对输入进行层归一化
        # 应用GELU激活函数并进行逐元素相乘
        x = F.gelu(self.Dconv1(x)) * self.Dconv2(x)
        # 残差连接
        x = x_o + self.conv(x)
        return x

# Restormer模型
class Restormer(nn.Module):
    def __init__(self, in_c, out_c):
        super(Restormer, self).__init__()
        # 1x1卷积，用于调整输入特征图的通道数
        self.mlp = nn.Conv2d(in_c, out_c, 1, 1, 0)
        # 多尺度动态注意力模块
        self.mdta = MDTA(out_c)
        # 门控深度前馈网络
        self.gdfn = GDFN(out_c)

    def forward(self, feature):
        feature = self.mlp(feature)  # 调整通道数
        feature = self.mdta(feature)  # 应用多尺度动态注意力模块
        return self.gdfn(feature)  # 应用门控深度前馈网络


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

class Bottleneck_Restormer(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Restormer(c_,c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_CGLU特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_Restormer(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_Restormer(self.c, self.c, shortcut, g) for _ in range(n)
        )



if __name__ == "__main__":

    # 随机生成输入特征图
    input_feature = torch.randn(8, 32, 27, 33)
    # 初始化Restormer模型
    model = Restormer(in_c=32, out_c=64)
    # 进行前向传播
    output = model(input_feature)
    print(f"输入特征图形状: {input_feature.shape}")
    print(f"输出特征图形状: {output.shape}")