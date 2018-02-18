# coding: utf-8

import chainer
from chainer import functions as F
import numpy


class LossCalculator:
    def __init__(self, n_classes, weight_noobj=0.2, weight_pos=1):
        self.xp = numpy
        self.class_weight = self.xp.ones(n_classes + 1, dtype=self.xp.float32)
        self.class_weight[0] = weight_noobj
        self.weight_pos = weight_pos
        self.n_classes = n_classes

    def to_gpu(self):
        import cupy
        self.xp = cupy
        self.class_weight_cpu = self.class_weight
        self.class_weight = chainer.cuda.to_gpu(self.class_weight_cpu)

    def to_cpu(self):
        self.xp = numpy
        self.class_weight = self.class_weight_cpu

    def loss(self, pred, actual):
        batch_size, n_boxes, _ = actual.shape
        id_array = self.xp.array(actual[:,:,4], dtype=self.xp.int32).reshape(batch_size * n_boxes)
        cl_pred = pred[:,:,4:].reshape((batch_size * n_boxes, self.n_classes + 1))
        cl_loss = F.softmax_cross_entropy(cl_pred, id_array, class_weight=self.class_weight)

        obj_idx = self.xp.where(actual[:,:,4] > 0)

        if obj_idx[0].size > 0:
            pred_boxes = pred[obj_idx][:,:4]
            actual_boxes = actual[obj_idx][:,:4]
            pos_loss = F.mean_squared_error(pred_boxes, actual_boxes)
            loss = cl_loss + self.weight_pos * pos_loss
        else:
            pos_loss = 0
            loss = cl_loss

        result = {
            'loss': loss,
            'pos_loss': pos_loss,
            'cl_loss': cl_loss,
            'cl_acc': F.accuracy(cl_pred, id_array)
        }
        return result
