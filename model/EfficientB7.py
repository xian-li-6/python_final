import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# EfficientNet 中的倒残差模块（MBConv Block）
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride

        # 扩展通道
        self.expand_channels = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # 1x1 卷积层 - 扩展通道
        self.conv1 = nn.Conv2d(in_channels, self.expand_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.expand_channels)

        # Depthwise 卷积层
        self.conv2 = nn.Conv2d(self.expand_channels, self.expand_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, groups=self.expand_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expand_channels)

        # SE模块
        self.se = SELayer(self.expand_channels, se_ratio) if se_ratio is not None else nn.Identity()

        # 1x1 卷积层 - 压缩通道
        self.conv3 = nn.Conv2d(self.expand_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

    def forward(self, x):
        residual = x

        # 扩展通道，激活函数，深度卷积
        x = self.swish(self.bn1(self.conv1(x)))
        x = self.swish(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.bn3(self.conv3(x))

        # 残差连接
        if self.use_res_connect:
            return x + residual
        else:
            return x

# EfficientNet 中的 Squeeze-and-Excitation (SE) 模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.swish = Swish()

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.swish(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

# EfficientNet-B7
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB7, self).__init__()

        # Stem部分
        self.conv_stem = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.swish = Swish()

        # 网络结构: 7个MBConvBlock阶段
        self.blocks = nn.ModuleList([
            MBConvBlock(64, 64, expand_ratio=1, stride=1, kernel_size=3),
            MBConvBlock(64, 128, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(128, 128, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(128, 256, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(256, 256, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(256, 512, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(512, 512, expand_ratio=6, stride=1, kernel_size=3),
            MBConvBlock(512, 1024, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(1024, 1024, expand_ratio=6, stride=1, kernel_size=3),
        ])

        # 分类器部分
        self.conv_head = nn.Conv2d(1024, 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.swish(self.bn0(self.conv_stem(x)))

        # MBConvBlock阶段
        for block in self.blocks:
            x = block(x)

        # 分类头
        x = self.swish(self.bn1(self.conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 创建 EfficientNet-B7 模型
model = EfficientNetB7(num_classes=1000)

# 打印模型结构
print(model)
