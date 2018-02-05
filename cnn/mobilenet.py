# coding: utf-8
# Original implementation from peisuke
# https://github.com/peisuke/DeepLearningSpeedComparison/blob/master/chainer/mobilenet/predict.py

import chainer
import chainer.functions as F
import chainer.links as L


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
            

class MobileNet(chainer.Chain):
    def __init__(self, n_classes=1000, n_base_units=32):
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
            self.conv_ds_14 = ConvDW(n_base_units *32, n_classes, 1)

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
 
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
        h = F.average_pooling_2d(h, 7, stride=1)
        h = F.reshape(h, (h.data.shape[0], h.data.shape[1]))
        return h
