# coding: utf-8

import chainer
from chainer import functions as F
from chainer import links as L

# relu6に差し替え予定
activation = F.relu


class ConvBN(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, pad=0):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize=ksize, stride=stride, pad=pad)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        return activation(self.bn(self.conv(x)))


class Bottleneck(chainer.Chain):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6, use_residual=False):
        super().__init__()
        with self.init_scope():
            self.conv1 = ConvBN(in_channels, in_channels * expansion_factor)
            self.dconv = L.DepthwiseConvolution2D(in_channels * expansion_factor, 1, ksize=3, stride=stride, pad=1)
            self.bn = L.BatchNormalization(in_channels * expansion_factor)
            self.conv2 = L.Convolution2D(in_channels * expansion_factor, out_channels, ksize=1, stride=1, pad=0)
        self.use_residual = use_residual

    def __call__(self, x):
        h = self.conv1(x)
        h = activation(self.bn(self.dconv(h)))
        h = self.conv2(h)
        if self.use_residual:
            h = h + x
        return h


class Bottleneck1(chainer.Chain):
    def __init__(self, in_channels, out_channels, expansion_factor=6):
        super().__init__()
        with self.init_scope():
            self.conv1 = ConvBN(in_channels, in_channels * expansion_factor)
            self.dconv = L.DepthwiseConvolution2D(in_channels * expansion_factor, 1, ksize=3, stride=1, pad=1)
            self.bn = L.BatchNormalization(in_channels * expansion_factor)
            self.conv2 = L.Convolution2D(in_channels * expansion_factor, out_channels, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        h = self.conv1(x)
        h = activation(self.bn(self.dconv(h)))
        h = self.conv2(h)
        return h + x

class Bottleneck2(chainer.Chain):
    def __init__(self, in_channels, out_channels, expansion_factor=6):
        super().__init__()
        with self.init_scope():
            self.conv1 = ConvBN(in_channels, in_channels * expansion_factor)
            self.dconv = L.DepthwiseConvolution2D(in_channels * expansion_factor, 1, ksize=3, stride=2, pad=1)
            self.bn = L.BatchNormalization(in_channels * expansion_factor)
            self.conv2 = L.Convolution2D(in_channels * expansion_factor, out_channels, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        h = self.conv1(x)
        h = activation(self.bn(self.dconv(h)))
        h = self.conv2(h)
        return h


class MobileNetV2(chainer.Chain):
    def __init__(self, n_classes, n_base_units=32, expansion_factor=6, dropout_prob=0.2):
        super().__init__()
        with self.init_scope():
            # 224 => 112
            self.conv01 = ConvBN(3, n_base_units, ksize=3, stride=2, pad=1)
            # 112 => 112
            self.conv02 = Bottleneck(n_base_units, max(3, int(n_base_units/2)), expansion_factor=1)
            # 112 => 56
            self.conv03 = Bottleneck(max(3, int(n_base_units/2)), max(3, int(n_base_units * 2 / 3)), stride=2, expansion_factor=expansion_factor)
            self.conv04 = Bottleneck(max(3, int(n_base_units * 2 / 3)), max(3, int(n_base_units * 2 / 3)), expansion_factor=expansion_factor, use_residual=True)
            # 56 => 28
            self.conv05 = Bottleneck(max(3, int(n_base_units * 2 / 3)), n_base_units, stride=2, expansion_factor=expansion_factor)
            self.conv06 = Bottleneck(n_base_units, n_base_units, expansion_factor=expansion_factor, use_residual=True)
            self.conv07 = Bottleneck(n_base_units, n_base_units, expansion_factor=expansion_factor, use_residual=True)
            # 28 => 14
            self.conv08 = Bottleneck(n_base_units, n_base_units * 2, stride=2, expansion_factor=expansion_factor)
            self.conv09 = Bottleneck(n_base_units * 2, n_base_units * 2, expansion_factor=expansion_factor, use_residual=True)
            self.conv10 = Bottleneck(n_base_units * 2, n_base_units * 2, expansion_factor=expansion_factor, use_residual=True)
            self.conv11 = Bottleneck(n_base_units * 2, n_base_units * 2, expansion_factor=expansion_factor, use_residual=True)
            # 14 => 14
            self.conv12 = Bottleneck(n_base_units * 2, n_base_units * 3, expansion_factor=expansion_factor)
            self.conv13 = Bottleneck(n_base_units * 3, n_base_units * 3, expansion_factor=expansion_factor, use_residual=True)
            self.conv14 = Bottleneck(n_base_units * 3, n_base_units * 3, expansion_factor=expansion_factor, use_residual=True)
            # 14 => 7
            self.conv15 = Bottleneck(n_base_units * 3, n_base_units * 5, stride=2, expansion_factor=expansion_factor)
            self.conv16 = Bottleneck(n_base_units * 5, n_base_units * 5, expansion_factor=expansion_factor, use_residual=True)
            self.conv17 = Bottleneck(n_base_units * 5, n_base_units * 5, expansion_factor=expansion_factor, use_residual=True)
            # 7 => 7
            self.conv18 = Bottleneck(n_base_units * 5, n_base_units * 10, expansion_factor=expansion_factor)
            # 7 => 7
            self.conv19 = L.Convolution2D(n_base_units * 10, n_base_units * 40, ksize=1, stride=1)
            # 1 => 1 Linearにしたほうがよいかも
            self.conv20 = L.Convolution2D(n_base_units * 40, n_classes, ksize=1, stride=1)
        self.dropout_prob = dropout_prob

    def predict(self, x):
        h = activation(self.conv01(x))
        h = self.conv02(h)
        h = self.conv03(h)
        h = self.conv04(h)
        h = self.conv05(h)
        h = self.conv06(h)
        h = self.conv07(h)
        h = self.conv08(h)
        h = self.conv09(h)
        h = self.conv10(h)
        h = self.conv11(h)
        h = self.conv12(h)
        h = self.conv13(h)
        h = self.conv14(h)
        h = self.conv15(h)
        h = self.conv16(h)
        h = self.conv17(h)
        h = self.conv18(h)
        h = self.conv19(h)
        h = F.average_pooling_2d(h, ksize=7)
        h = F.dropout(h, self.dropout_prob)
        h = self.conv20(h)
        h = F.reshape(h, (h.data.shape[0], h.data.shape[1]))
        return h

    def __call__(self, x, t=None):
        pred = self.predict(x)
        if t is None:
            return pred

        loss = F.softmax_cross_entropy(pred, t)
        acc = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss
