import torch.nn as nn
import torch


class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops

class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, kernel_size, stride, act=mv_act, out_dim=dim)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim, kernel_size, stride, act=mv_act, out_dim=dim,
                                      exp_factor=2., expand_groups=1)

    def forward(self, attn_list):
        # for i, attn in enumerate(attn_list[:-1]):
        #     attn = F.pixel_shuffle(attn, 2 ** i)
        #     x = attn if i == 0 else torch.cat([x, attn], dim=1)
        # x = torch.cat([attn_list[0], attn_list[1]], dim=1) # 将这个contact放在外面
        x = self.mobivari(attn_list)
        return x

    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)


if __name__ == '__main__':

    hrami = HRAMi(dim=64)
    input = [
        torch.randn(1, 64, 32, 32),  # Level 0
        torch.randn(1, 32, 32, 32)  # Level 3 (final level)
    ]
    output = hrami(input)

    print(output.size())