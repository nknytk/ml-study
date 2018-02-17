# coding: utf-8

import chainer
from chainer import functions as F
import numpy


class LossCalculator:
    def __init__(self, n_classes, weight_noobj=0.2, weight_pos=1):
        self.class_weight = numpy.ones(n_classes + 1, dtype=numpy.float32)
        self.class_weight[0] = weight_noobj
        self.weight_pos = weight_pos
        self.n_classes = n_classes

    def loss(self, pred, actual):
        batch_size, n_boxes, _ = actual.shape
        id_array = numpy.array(actual[:,:,4], dtype=numpy.int32).reshape(batch_size * n_boxes)
        cl_pred = pred[:,:,4:].reshape((batch_size * n_boxes, self.n_classes + 1))
        cl_loss = F.softmax_cross_entropy(cl_pred, id_array, class_weight=self.class_weight)

        obj_idx = numpy.where(actual[:,:,4] > 0)

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
