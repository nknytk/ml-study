# coding: utf-8
# original implementation by chainer
# https://github.com/chainer/chainer/blob/master/examples/imagenet/googlenetbn.py

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self, n_classes=1000, n=32):
        super(GoogLeNetBN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n * 2, 7, stride=2, pad=3, nobias=True)
            self.norm1 = L.BatchNormalization(n * 2)
            self.conv2 = L.Convolution2D(None, n * 6, 3, pad=1, nobias=True)
            self.norm2 = L.BatchNormalization(n * 6)
            self.inc3a = L.InceptionBN(
                None, n * 2, n * 2, n * 2, n * 2, n * 3, 'avg', n)
            self.inc3b = L.InceptionBN(
                None, n * 2, n * 2, n * 3, n * 2, n * 3, 'avg', n * 1)
            self.inc3c = L.InceptionBN(
                None, 0, n * 4, n * 5, n * 2, n * 3, 'max', stride=2)
            self.inc4a = L.InceptionBN(
                None, n * 7, n * 2, n * 3, n * 3, n * 4, 'avg', n * 4)
            self.inc4b = L.InceptionBN(
                None, n * 6, n * 3, n * 4, n * 3, n * 4, 'avg', n * 4)
            self.inc4c = L.InceptionBN(
                None, n * 5, n * 4, n * 5, n * 4, n * 5, 'avg', n * 4)
            self.inc4d = L.InceptionBN(
                None, n * 3, n * 4, n * 6, n * 5, n * 6, 'avg', n * 4)
            self.inc4e = L.InceptionBN(
                None, 0, n * 4, n * 6, n * 6, n * 8, 'max', stride=2)
            self.inc5a = L.InceptionBN(
                None, n * 11, n * 6, n * 10, n * 5, n * 7, 'avg', n * 4)
            self.inc5b = L.InceptionBN(
                None, n * 11, n * 6, n * 10, n * 6, n * 7, 'max', n * 4)
            self.out = L.Linear(None, n_classes)

            self.conva = L.Convolution2D(None, n * 4, 1, nobias=True)
            self.norma = L.BatchNormalization(n * 4)
            self.lina = L.Linear(None, n * 32, nobias=True)
            self.norma2 = L.BatchNormalization(n * 32)
            self.outa = L.Linear(None, n_classes)

            self.convb = L.Convolution2D(None, n * 4, 1, nobias=True)
            self.normb = L.BatchNormalization(n * 4)
            self.linb = L.Linear(None, n * 32, nobias=True)
            self.normb2 = L.BatchNormalization(n * 32)
            self.outb = L.Linear(None, n_classes)

    def __call__(self, x, t):
        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a)))
        a = F.relu(self.norma2(self.lina(a)))
        a = self.outa(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b)))
        b = F.relu(self.normb2(self.linb(b)))
        b = self.outb(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
        }, self)
        return loss

    def predict(self, x):
        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)

        return h
