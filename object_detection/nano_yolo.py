# coding: utf-8

import os
import sys
import chainer
from chainer import links as L
from chainer import functions as F

sys.path.append(os.path.dirname(__file__))
from loss import LossCalculator


class NanoYOLO(chainer.Chain):
    """
    Input: (3, 100, 100)の画像
    Output: (n_boxes, 5 + n_classes)
      画像を3x3のタイルに分け、1つずつ
      オブジェクト中心点x, y, オブジェクト大きさx, y, IOU, クラス確率 * n_classes
      のベクトルを出力
    """
    def __init__(self, n_classes, n_base_units=16, n_boxes=5):
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
            # input: (n_base_units * 4, 9, 9)
            conv4 = L.Convolution2D(n_base_units * 4, n_base_units * 8, 3, stride=1, pad=1),
            bn4 = L.BatchNormalization(n_base_units * 8),
            # input: (n_base_units * 8, 1, 1)
            fc=L.Linear(n_base_units * 8, n_boxes * (5 + n_classes))
        )
        self.feature_dim = n_base_units * 8
        self.loss_calc = LossCalculator(n_classes)
        self.n_classes = n_classes
        self.n_boxes = n_boxes

    def __call__(self, x, t):
        pred = self.predict(x)
        evaluated = self.loss_calc.loss(pred, t)
        chainer.report(evaluated, self)
        return evaluated['loss']

    def predict(self, x):
        h = self.reduct(x)
        h = self.fc(h)

        batch_size = x.data.shape[0]
        r = F.reshape(h, (batch_size, self.n_boxes, 5 + self.n_classes))
        return r

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


class MicroYOLO(chainer.Chain):
    def __init__(self, n_classes, n_base_units=16, n_boxes=5):
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
            fc=L.Linear(n_base_units * 18, n_boxes * (5 + n_classes))
        )
        self.feature_dim = n_base_units * 8
        self.loss_calc = LossCalculator(n_classes)
        self.n_classes = n_classes
        self.n_boxes = n_boxes

    def __call__(self, x, t):
        pred = self.predict(x)
        evaluated = self.loss_calc.loss(pred, t)
        chainer.report(evaluated, self)
        return evaluated['loss']

    def predict(self, x):
        h = self.reduct(x)
        h = self.fc(h)

        batch_size = x.data.shape[0]
        r = F.reshape(h, (batch_size, self.n_boxes, 5 + self.n_classes))
        return r

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
