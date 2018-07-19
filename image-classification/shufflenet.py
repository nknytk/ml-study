# coding: utf-8

import chainer
from chainer import functions as F
from chainer import links as L


def shuffle_channels(x, num_groups):
    batch_size, channel, height, width = x.shape
    x = F.reshape(x, (batch_size, num_groups, channel // num_groups, height, width))
    x = F.transpose(x, (0, 2, 1, 3, 4))
    return F.reshape(x, (batch_size, channel, height, width))


class Stride1Block(chainer.Chain):
    def __init__(self, channels, num_groups):
        bottleneck_channels = int(channels/4)
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(channels, bottleneck_channels, ksize=1, stride=1, pad=0, groups=num_groups)
            self.bn1 = L.BatchNormalization(bottleneck_channels)
            self.conv2 = L.DepthwiseConvolution2D(bottleneck_channels, 1, ksize=3, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(bottleneck_channels)
            self.conv3 = L.Convolution2D(bottleneck_channels, channels, ksize=1, stride=1, pad=0, groups=num_groups)
            self.bn3 = L.BatchNormalization(channels)
        self.num_groups = num_groups

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = shuffle_channels(h, self.num_groups)
        h = self.bn2(self.conv2(h))
        h = self.bn3(self.conv3(h))
        return h + x


class Stride2Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, num_groups):
        bottleneck_channels = int((out_channels)/4)
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, bottleneck_channels, ksize=1, stride=1, pad=0, groups=num_groups)
            self.bn1 = L.BatchNormalization(bottleneck_channels)
            self.conv2 = L.DepthwiseConvolution2D(bottleneck_channels, 1, ksize=3, stride=2, pad=1)
            self.bn2 = L.BatchNormalization(bottleneck_channels)
            self.conv3 = L.Convolution2D(bottleneck_channels, out_channels - in_channels, ksize=1, stride=1, pad=0, groups=num_groups)
            self.bn3 = L.BatchNormalization(out_channels - in_channels)
        self.num_groups = num_groups

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = shuffle_channels(h, self.num_groups)
        h = self.bn2(self.conv2(h))
        h = self.bn3(self.conv3(h))
 
        h2 = F.average_pooling_2d(x, ksize=3, stride=2, pad=1)
        return F.concat((h, h2))


class ShuffleNetG8(chainer.Chain):
    def __init__(self, n_classes, scale_factor=24, dr_proba=0):
        super().__init__()
        with self.init_scope():
            # (3, 224, 224) => (scale_factor, 112, 112)
            self.conv1 = L.Convolution2D(3, scale_factor, ksize=3, stride=2, pad=1)
            self.bn1 = L.BatchNormalization(scale_factor)
            # (scale_factor, 112, 112) => pooling => (scale_factor, 56, 56)
            # (scale_factor, 56, 56) => (scale_factor * 16, 28, 28)
            self.stage2_1 = Stride2Block(scale_factor, scale_factor * 16, 8)
            self.stage2_2 = Stride1Block(scale_factor * 16, 8)
            self.stage2_3 = Stride1Block(scale_factor * 16, 8)
            self.stage2_4 = Stride1Block(scale_factor * 16, 8)
            # (scale_factor * 16, 28, 28) => (scale_factor * 32, 14, 14)
            self.stage3_1 = Stride2Block(scale_factor * 16, scale_factor * 32, 8)
            self.stage3_2 = Stride1Block(scale_factor * 32, 8)
            self.stage3_3 = Stride1Block(scale_factor * 32, 8)
            self.stage3_4 = Stride1Block(scale_factor * 32, 8)
            self.stage3_5 = Stride1Block(scale_factor * 32, 8)
            self.stage3_6 = Stride1Block(scale_factor * 32, 8)
            self.stage3_7 = Stride1Block(scale_factor * 32, 8)
            self.stage3_8 = Stride1Block(scale_factor * 32, 8)
            # (scale_factor * 32, 14, 14) => (scale_factor * 64, 7, 7)
            self.stage4_1 = Stride2Block(scale_factor * 32, scale_factor * 64, 8)
            self.stage4_2 = Stride1Block(scale_factor * 64, 8)
            self.stage4_3 = Stride1Block(scale_factor * 64, 8)
            self.stage4_4 = Stride1Block(scale_factor * 64, 8)
            self.fc = L.Linear(scale_factor * 64, n_classes)
        self.dr_proba = dr_proba

    def predict(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, ksize=3, stride=2, pad=1)
        h = self.stage2_1(h)
        h = self.stage2_2(h)
        h = self.stage2_3(h)
        h = self.stage2_4(h)
        h = self.stage3_1(h)
        h = self.stage3_2(h)
        h = self.stage3_3(h)
        h = self.stage3_4(h)
        h = self.stage3_5(h)
        h = self.stage3_6(h)
        h = self.stage3_7(h)
        h = self.stage3_8(h)
        h = self.stage4_1(h)
        h = self.stage4_2(h)
        h = self.stage4_3(h)
        h = self.stage4_4(h)
        h = F.average_pooling_2d(h, ksize=7)
        if self.dr_proba:
            h = F.dropout(h, self.dr_proba)
        return self.fc(h)

    def __call__(self, x, t=None):
        pred = self.predict(x)
        if t is None:
            return pred

        loss = F.softmax_cross_entropy(pred, t)
        acc = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss
