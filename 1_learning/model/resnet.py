import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义ResNet的基础残差块（BasicBlock）
class BasicBlock(nn.Module):
    expansion = 1  # 通道扩展系数，BasicBlock为1，Bottleneck为4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样（用于匹配通道数/尺寸）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存

    def forward(self, x):
        residual = x  # 残差连接
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 下采样匹配残差的通道/尺寸
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual  # 残差相加
        out = self.relu(out)
        
        return out

# 定义轻量化ResNet（适配笔记本，基于ResNet18简化）
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=27):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始通道数
        
        # 初始卷积层（简化，适配224x224输入）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层（4层，layers=[2,2,2,2]对应ResNet18，轻量化）
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化（替代flatten，减少参数和内存）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类头（适配27分类）
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重（提升收敛速度）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建残差层"""
        downsample = None
        # 需要下采样的情况：步长≠1 或 输入通道≠输出通道×扩展系数
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 池化+分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 快捷函数：创建ResNet18（轻量化，适配笔记本）
def ResNet18(num_classes=27):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# 如果你需要更轻量的版本（ResNet8，仅适配小数据集）
def ResNet8(num_classes=27):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)