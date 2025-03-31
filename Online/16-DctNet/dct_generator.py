import mindspore.ops as ops
import mindspore.nn as nn


# 定义残差块
class ResidualBlock(nn.Cell):
    def __init__(self, channels, kernel_size, stride, padding, pad_mode):
        super().__init__()
        # 定义残差块的卷积层和激活函数
        self.block = nn.SequentialCell(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, has_bias=True),
        )

    def construct(self, x):
        # 返回输入和残差块的输出相加
        return x + self.block(x)

# 定义生成器
class Generator(nn.Cell):
    def __init__(self, img_channels=3, num_features=32, num_residuals=4, pad_mode="pad"):
        super().__init__()
        self.pad_mode = pad_mode
        # 定义生成器的初始下采样层
        self.initial_down = nn.SequentialCell(
            # k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, pad_mode=self.pad_mode, padding=3, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )
        #Down-convolution
        self.down1 = nn.SequentialCell(
            # k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),

            #k3n64s1
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.down2 = nn.SequentialCell(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),

            #k3n128s1
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]
        self.res_blocks = nn.SequentialCell(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up1 = nn.SequentialCell(
            #k3n128s1
            nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.up2 = nn.SequentialCell(
            #k3n64s1
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            #k3n64s1
            nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.last = nn.SequentialCell(
            #k3n32s1
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            #k7n3s1
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, pad_mode=self.pad_mode, padding=3, has_bias=True)
        )

    def construct(self, x):
        x1 = self.initial_down(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.res_blocks(x)
        x = self.up1(x)

        # 这里源代码使用了双线性插值bilinear，但MindSpore的双线性插值函数不支持scale_factor，因此这里使用area插值，差别不明显
        # 当然，为了追求效果，也可以使用双线性插值from scipy.ndimage import zoom，但这样会导致Generator无法使用静态图从而影响性能
        x = ops.interpolate(x, scale_factor=2.0, mode='area')
        x = self.up2(x + x2)
        x = ops.interpolate(x, scale_factor=2.0, mode='area')
        x = self.last(x + x1)
        # TanH
        return ops.tanh(x)
