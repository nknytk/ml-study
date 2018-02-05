# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
            bn=L.BatchNormalization(e1 + e3)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat((h_1, h_3), axis=1)
        return F.relu(self.bn(h_out))


class SqueezeNet(chainer.Chain):
    def __init__(self, dim_out=1000, base_filter_num=16):
        super().__init__(
            conv1=L.Convolution2D(3, base_filter_num * 6, 7, stride=2),
            fire2=Fire(base_filter_num * 6, base_filter_num, base_filter_num * 4, base_filter_num * 4),
            fire3=Fire(base_filter_num * 8, base_filter_num, base_filter_num * 4, base_filter_num * 4),
            fire4=Fire(base_filter_num * 8, base_filter_num * 2, base_filter_num * 8, base_filter_num * 8),
            fire5=Fire(base_filter_num * 16, base_filter_num * 2, base_filter_num * 8, base_filter_num * 8),
            fire6=Fire(base_filter_num * 16, base_filter_num * 3, base_filter_num * 12, base_filter_num * 12),
            fire7=Fire(base_filter_num * 24, base_filter_num * 3, base_filter_num * 12, base_filter_num * 12),
            fire8=Fire(base_filter_num * 24, base_filter_num * 4, base_filter_num * 16, base_filter_num * 16),
            fire9=Fire(base_filter_num * 32, base_filter_num * 4, base_filter_num * 16, base_filter_num * 16),
            conv10=L.Convolution2D(base_filter_num * 32, dim_out, 1, pad=1)
        )
        self.dim_out = dim_out

    def __call__(self, x, t):
        pred = self.predict(x)
        loss = F.softmax_cross_entropy(pred, t)
        acc = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss

    def predict(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = self.fire8(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire9(h)
        h = F.dropout(h, ratio=0.5)

        h = F.relu(self.conv10(h))
        h = F.average_pooling_2d(h, 13)

        return F.reshape(h, (-1, self.dim_out))
