import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：A Lightweight Fusion Strategy With Enhanced Interlayer Feature Correlation for Small Object Detection
# 论文链接：https://ieeexplore.ieee.org/abstract/document/10671587
# 官方github：https://github.com/nuliweixiao/EFC


class EFC_block(nn.Module):
    def __init__(self, c1):
        super().__init__()
        # 定义卷积层
        c2=c1
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)  # 1x1卷积，输入通道c1，输出通道c2
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)  # 1x1卷积，输入输出通道均为c2
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)  # 1x1卷积，输入输出通道均为c2
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层，通道数为c2
        self.sigomid = nn.Sigmoid()  # Sigmoid激活函数
        self.group_num = 16  # 分组数
        self.eps = 1e-10  # 极小值，用于数值稳定性
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))  # 可学习的偏移参数
        # 门控生成器，用于生成权重
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(c2, c2, 1, 1),  # 1x1卷积
            nn.ReLU(True),  # ReLU激活函数
            nn.Softmax(dim=1),  # Softmax归一化
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)  # 深度可分离卷积
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)  # 1x1卷积
        self.Apt = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.one = c2  # 通道数
        self.two = c2  # 通道数
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)  # 1x1卷积，输出通道为1
        # 定义4个交互卷积层
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1)  # 1x1卷积，输入输出通道为c2//4

        self.conv_match = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.conv_down = nn.Conv2d(256, 384, kernel_size=1, stride=1)


    def forward(self, x):
        x1, x2 = x  # 解包输入
        x2 = self.conv_match(x2)

        global_conv1 = self.conv1(x1)  # 对x1进行1x1卷积
        bn_x = self.bn(global_conv1)  # 批归一化
        weight_1 = self.sigomid(bn_x)  # Sigmoid激活

        global_conv2 = self.conv2(x2)  # 对x2进行1x1卷积
        bn_x2 = self.bn(global_conv2)  # 批归一化
        weight_2 = self.sigomid(bn_x2)  # Sigmoid激活

        X_GOBAL = global_conv1 + global_conv2  # 特征融合
        x_conv4 = self.conv4_gobal(X_GOBAL)  # 对融合后的特征进行1x1卷积
        X_4_sigmoid = self.sigomid(x_conv4)  # Sigmoid激活

        # GFF
        X_ = X_4_sigmoid * X_GOBAL  # 加权融合
        X_ = X_.chunk(4, dim=1)  # 将特征分成4组
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])  # 对每组特征进行1x1卷积
            N, C, H, W = out_1.size()  # 获取特征图的形状
            x_1_map = out_1.reshape(N, 1, -1)  # 重塑特征图
            mean_1 = x_1_map.mean(dim=2, keepdim=True)  # 计算均值
            x_1_av = x_1_map / mean_1  # 归一化
            x_2_2 = F.softmax(x_1_av, dim=1)  # Softmax归一化
            x1 = x_2_2.reshape(N, C, H, W)  # 重塑回原形状
            x1 = X_[group_id] * x1  # 加权
            out.append(x1)  # 添加到输出列表
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)  # 拼接4组特征

        #MFR
        N, C, H, W = out.size()  # 获取特征图的形状
        x_add_1 = out.reshape(N, self.group_num, -1)  # 重塑特征图
        N, C, H, W = X_GOBAL.size()  # 获取全局特征的形状
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)  # 重塑全局特征
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)  # 计算均值
        std_1 = x_shape_1.std(dim=2, keepdim=True)  # 计算标准差
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)  # 归一化
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)  # 重塑回原形状
        x_gui = (x_guiyi_1 * self.gamma + self.beta)  # 缩放和偏移

        weight_x3 = self.Apt(X_GOBAL)  # 全局平均池化
        reweights = self.sigomid(weight_x3)  # Sigmoid激活
        x_up_1 = reweights >= weight_1  # 判断条件
        x_low_1 = reweights < weight_1  # 判断条件
        x_up_2 = reweights >= weight_2  # 判断条件
        x_low_2 = reweights < weight_2  # 判断条件

        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL  # 加权融合
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL  # 加权融合
        x11_up_dwc = self.dwconv(x_low)  # 深度可分离卷积
        x11_up_dwc = self.conv3(x11_up_dwc)  # 1x1卷积
        x_so = self.gate_genator(x_low)  # 生成门控权重
        x11_up_dwc = x11_up_dwc * x_so  # 加权
        x22_low_pw = self.conv4(x_up)  # 1x1卷积
        xL = x11_up_dwc + x22_low_pw  # 特征融合
        xL = xL + x_gui  # 添加归一化后的特征

        xL = self.conv_down(xL)

        return xL  # 返回输出

if __name__ == '__main__':
    x1 = torch.randn(1, 256, 64, 64)  # 随机生成输入特征图1
    x2 = torch.randn(1, 128, 64, 64)  # 随机生成输入特征图2
    x = (x1, x2)  # 打包输入
    model = EFC_block(256)  # 实例化模型
    print(model(x).shape)  # 打印输出特征图的形状