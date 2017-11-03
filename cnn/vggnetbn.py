# coding: utf-8

import chainer
from  chainer import links as L
from chainer import functions as F


class VGGNetBN(chainer.Chain):

    """ VGGNetを元に、収束を早くするためBatchNormalizationを追加 """

    def __init__(self, dim_out, n_base_filters=64, dropout=0.5):
        super().__init__(
            conv1_1=L.Convolution2D(3, n_base_filters, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(n_base_filters, n_base_filters, 3, stride=1, pad=1),
            bn1=L.BatchNormalization(n_base_filters),

            conv2_1=L.Convolution2D(n_base_filters, n_base_filters * 2, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(n_base_filters * 2, n_base_filters * 2, 3, stride=1, pad=1),
            bn2=L.BatchNormalization(n_base_filters * 2),

            conv3_1=L.Convolution2D(n_base_filters * 2, n_base_filters * 4, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(n_base_filters * 4, n_base_filters * 4, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(n_base_filters * 4, n_base_filters * 4, 3, stride=1, pad=1),
            bn3=L.BatchNormalization(n_base_filters * 4),

            conv4_1=L.Convolution2D(n_base_filters * 4, n_base_filters * 8, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(n_base_filters * 8, n_base_filters * 8, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(n_base_filters * 8, n_base_filters * 8, 3, stride=1, pad=1),
            bn4=L.BatchNormalization(n_base_filters * 8),

            conv5_1=L.Convolution2D(n_base_filters * 8, n_base_filters * 8, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(n_base_filters * 8, n_base_filters * 8, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(n_base_filters * 8, n_base_filters * 8, 3, stride=1, pad=1),
            bn5=L.BatchNormalization(n_base_filters * 8),

            fc6=L.Linear(n_base_filters * 392, n_base_filters * 64),
            fc7=L.Linear(n_base_filters * 64, n_base_filters * 64),
            fc8=L.Linear(n_base_filters * 64, dim_out)
        )
        self.dropout_rate = dropout

    def __call__(self, x, t):
        pred = self.predict(x)
        loss = F.softmax_cross_entropy(pred, t)
        acc = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss

    def predict(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.bn1(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.bn2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.bn3(self.conv3_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.bn4(self.conv4_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.bn5(self.conv5_3(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), self.dropout_rate)
        h = F.dropout(F.relu(self.fc7(h)), self.dropout_rate)
        return self.fc8(h)
