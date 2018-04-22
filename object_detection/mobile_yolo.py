# coding: utf-8
# Original implementation from peisuke
# https://github.com/peisuke/DeepLearningSpeedComparison/blob/master/chainer/mobilenet/predict.py

import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L

sys.path.append(os.path.dirname(__file__))
from loss import LossCalculator


class ConvBN(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv=L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True)
            self.bn=L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        return h


class ConvDW(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        with self.init_scope():
            self.conv_dw=L.DepthwiseConvolution2D(inp, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw=L.BatchNormalization(inp)
            self.conv_sep=L.Convolution2D(inp, oup, 1, stride=1, pad=0, nobias=True)
            self.bn_sep=L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn_dw(self.conv_dw(x)))
        h = F.relu(self.bn_sep(self.conv_sep(h)))
        return h
            

class MobileYOLO(chainer.Chain):

    img_size = 224
    n_grid=7

    def __init__(self, n_classes=1, n_base_units=32, class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        with self.init_scope():
            self.conv_bn = ConvBN(3, n_base_units, 2)
            self.conv_ds_2 = ConvDW(n_base_units, n_base_units * 2, 1)
            self.conv_ds_3 = ConvDW(n_base_units * 2, n_base_units * 4, 2)
            self.conv_ds_4 = ConvDW(n_base_units * 4, n_base_units * 4, 1)
            self.conv_ds_5 = ConvDW(n_base_units * 4, n_base_units * 8, 2)
            self.conv_ds_6 = ConvDW(n_base_units * 8, n_base_units * 8, 1)
            self.conv_ds_7 = ConvDW(n_base_units * 8, n_base_units *16, 2)

            self.conv_ds_8 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_9 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_10 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_11 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_12 = ConvDW(n_base_units *16, n_base_units *16, 1)

            self.conv_ds_13 = ConvDW(n_base_units *16, n_base_units *32, 2)
            self.conv_ds_14 = ConvDW(n_base_units *32, 4 + n_classes, 1)

        self.loss_calc = LossCalculator(n_classes, weight_noobj=0.2, class_weights=class_weights)


    def __call__(self, x, t):
        pred = self.predict(x)
        evaluated = self.loss_calc.loss(pred, t)
        chainer.report(evaluated, self)
        return evaluated['loss']
 
    def predict(self, x):
        h = self.conv_bn(x)
        h = self.conv_ds_2(h)
        h = self.conv_ds_3(h)
        h = self.conv_ds_4(h)
        h = self.conv_ds_5(h)
        h = self.conv_ds_6(h)
        h = self.conv_ds_7(h)
        h = self.conv_ds_8(h)
        h = self.conv_ds_9(h)
        h = self.conv_ds_10(h)
        h = self.conv_ds_11(h)
        h = self.conv_ds_12(h)
        h = self.conv_ds_13(h)
        h = self.conv_ds_14(h)

        # (batch_size, 4 + n_classes, 7, 7) -> (bach_size, 7, 7, 4 + n_classes)
        h = F.transpose(h, (0, 2, 3, 1))
        # (batch_size, 7, 7, 4 + n_classes) -> (batch_size, 49, 4 + n_classes)
        batch_size = int(h.size / (self.n_grid**2 * (4 + self.n_classes)))
        r = F.reshape(h, (batch_size, self.n_grid**2, 4 + self.n_classes))
        return r

    def to_gpu(self, *args, **kwargs):
        self.loss_calc.to_gpu()
        return super().to_gpu(*args, **kwargs)

    def to_cpu(self, *args, **kwargs):
        self.loss_calc.to_cpu()
        return super().to_gpu(*args, **kwargs)


class MicroYOLO(chainer.Chain):

    img_size = 224
    n_grid=7

    def __init__(self, n_classes=1, n_base_units=32, class_weights=None):
        super().__init__()
        self.n_classes = n_classes
        with self.init_scope():
            self.conv_bn = ConvBN(3, n_base_units, 2)
            self.conv_ds_2 = ConvDW(n_base_units, n_base_units * 2, 1)
            self.conv_ds_3 = ConvDW(n_base_units * 2, n_base_units * 4, 2)
            self.conv_ds_4 = ConvDW(n_base_units * 4, n_base_units * 4, 1)
            self.conv_ds_5 = ConvDW(n_base_units * 4, n_base_units * 8, 2)
            self.conv_ds_6 = ConvDW(n_base_units * 8, n_base_units * 8, 1)
            self.conv_ds_7 = ConvDW(n_base_units * 8, n_base_units *16, 2)

            self.conv_ds_8 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_9 = ConvDW(n_base_units *16, n_base_units *16, 1)
            self.conv_ds_10 = ConvDW(n_base_units *16, n_base_units *16, 1)

            self.conv_ds_11 = ConvDW(n_base_units *16, n_base_units *32, 2)
            self.conv_ds_12 = ConvDW(n_base_units *32, 4 + n_classes, 1)

        self.loss_calc = LossCalculator(n_classes, weight_noobj=0.2, class_weights=class_weights)

    def __call__(self, x, t):
        pred = self.predict(x)
        evaluated = self.loss_calc.loss(pred, t)
        chainer.report(evaluated, self)
        return evaluated['loss']

    def predict(self, x):
        h = self.conv_bn(x)
        h = self.conv_ds_2(h)
        h = self.conv_ds_3(h)
        h = self.conv_ds_4(h)
        h = self.conv_ds_5(h)
        h = self.conv_ds_6(h)
        h = self.conv_ds_7(h)
        h = self.conv_ds_8(h)
        h = self.conv_ds_9(h)
        h = self.conv_ds_10(h)
        h = self.conv_ds_11(h)
        h = self.conv_ds_12(h)

        # (batch_size, 4 + n_classes, 7, 7) -> (bach_size, 7, 7, 4 + n_classes)
        h = F.transpose(h, (0, 2, 3, 1))
        # (batch_size, 7, 7, 4 + n_classes) -> (batch_size, 49, 4 + n_classes)
        batch_size = int(h.size / (self.n_grid**2 * (4 + self.n_classes)))
        r = F.reshape(h, (batch_size, self.n_grid**2, 4 + self.n_classes))
        return r

    def to_gpu(self, *args, **kwargs):
        self.loss_calc.to_gpu()
        return super().to_gpu(*args, **kwargs)

    def to_cpu(self, *args, **kwargs):
        self.loss_calc.to_cpu()
        return super().to_gpu(*args, **kwargs)
