import torch
import torch.nn as nn


# 论文题目：MixDehazeNet : Mix Structure Block For Image Dehazing Network
# 论文链接：https://ieeexplore.ieee.org/document/10651326
# 官方github：https://github.com/AmeryXiong/MixDehazeNet
# 代码改进者：一勺汤

class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        # self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        # self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        # self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


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

class Bottleneck_MixStructureBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = MixStructureBlock(c_)
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
        self.m = nn.Sequential(*(Bottleneck_MixStructureBlock(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_MixStructureBlock(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



import torch

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 定义输入张量的形状 (batch_size, channels, height, width)

    # 创建一个随机输入张量
    x = torch.randn(4, 64, 27, 32)

    # 打印输入张量的形状
    print(f"Input shape: {x.shape}")

    # 初始化 MixStructureBlock 模块
    mix_structure_block = MixStructureBlock(dim=64)

    # 前向传播
    output = mix_structure_block(x)

    # 打印输出张量的形状
    print(f"Output shape: {output.shape}")

    # 检查输入和输出形状是否一致
    assert x.shape == output.shape, "Input and output shapes do not match!"

    print("MixStructureBlock forward pass successful!")

if __name__ == "__main__":
    main()










