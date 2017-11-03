# coding: utf-8

import chainer
from chainer import links as L
from chainer import functions as F


class FaceClassifier100x100V2(chainer.Chain):
    """
    100x100の顔画像を分類する、VGGNetを参考にした実装。以下の処理を繰り返し、最後に全結合
      convolution1: サイズ無変更でchannelを増やす
      convolution1: サイズ,channelともに無変更
      batch normalization
      max pooling: サイズを半分にする
    """
    def __init__(self, n_classes, n_base_units=16):
        super().__init__(
            # input: (3, 100, 100)
            conv1_1 = L.Convolution2D(3, n_base_units, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D(n_base_units, n_base_units, 3, stride=1, pad=1),
            bn1 = L.BatchNormalization(n_base_units),
            # input: (n_base_units, 50, 50)
            conv2_1 = L.Convolution2D(n_base_units, n_base_units * 2, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(n_base_units * 2, n_base_units * 2, 3, stride=1, pad=1),
            bn2 = L.BatchNormalization(n_base_units * 2),
            # input: (n_base_units * 2, 25, 25)
            conv3_1 = L.Convolution2D(n_base_units * 2, n_base_units * 6, 3, stride=1, pad=1),
            conv3_2 = L.Convolution2D(n_base_units * 6, n_base_units * 6, 3, stride=1, pad=1),
            bn3 = L.BatchNormalization(n_base_units * 6),
            # input: (n_base_units * 4, 9, 9)
            conv4_1 = L.Convolution2D(n_base_units * 6, n_base_units * 18, 3, stride=1, pad=1),
            conv4_2 = L.Convolution2D(n_base_units * 18, n_base_units * 18, 3, stride=1, pad=1),
            bn4 = L.BatchNormalization(n_base_units * 18),
            # input: (n_base_units * 16, 1, 1)
            fc=L.Linear(n_base_units * 18, n_classes)
        )
        self.n_units = n_base_units * 18

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.reduct(x)
        h = self.fc(F.dropout(h, 0.4))
        return h

    def reduct(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.bn1(self.conv1_2(h)))
        # 100 -> 50
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.bn2(self.conv2_2(h)))
        # 50 -> 25
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.bn3(self.conv3_2(h)))
        # 25 -> (25 + 1 * 2 - 3) / 3 + 1 = 9
        h = F.max_pooling_2d(h, 3, stride=3, pad=1)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.bn4(self.conv4_2(h)))
        # 9 -> 1
        h = F.average_pooling_2d(h, 9, stride=1)

        return h

class FaceClassifier100x100V(chainer.Chain):
    """
    100x100の顔画像を分類する、VGGNetを参考にした実装。以下の処理を繰り返し、最後に全結合を一層だけ実施
      convolution: サイズ無変更でchannelを増やす
      batch normalization
      pooling: サイズを縮小
    """
    def __init__(self, n_classes, n_base_units=16):
        super().__init__(
            # input: (3, 100, 100)
            conv1_1 = L.Convolution2D(3, n_base_units, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D(n_base_units, n_base_units, 3, stride=1, pad=1),
            bn1 = L.BatchNormalization(n_base_units),
            # input: (n_base_units, 50, 50)
            conv2 = L.Convolution2D(n_base_units, n_base_units * 2, 3, stride=1, pad=1),
            bn2 = L.BatchNormalization(n_base_units * 2),
            # input: (n_base_units * 2, 25, 25)
            conv3 = L.Convolution2D(n_base_units * 2, n_base_units * 4, 3, stride=1, pad=1),
            bn3 = L.BatchNormalization(n_base_units * 4),
            # input: (n_base_units * 6, 9, 9)
            conv4 = L.Convolution2D(n_base_units * 4, n_base_units * 8, 3, stride=1, pad=1),
            bn4 = L.BatchNormalization(n_base_units * 8),
            # input: (n_base_units * 18, 1, 1)
            fc=L.Linear(n_base_units * 8, n_classes)
        )
        self.n_units = n_base_units * 8

    def __call__(self, x, t):
        h = self.predict(x)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.reduct(x)
        h = self.fc(F.dropout(h, 0.4))
        return h

    def reduct(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = self.bn1(h)
        # 100 -> 50
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2(h))
        h = self.bn2(h)
        # 50 -> 25
        h = F.max_pooling_2d(h, 2, stride=2)


        h = F.relu(self.conv3(h))
        h = self.bn3(h)
        # 25 -> (25 + 1 * 2 - 3) / 3 + 1 = 9
        h = F.max_pooling_2d(h, 3, stride=3, pad=1)

        h = F.relu(self.conv4(h))
        h = self.bn4(h)
        # 9 -> 1
        h = F.average_pooling_2d(h, 9, stride=1)

        return h
