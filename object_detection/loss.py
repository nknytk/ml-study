# coding: utf-8

import chainer
from chainer import functions as F
import numpy


class LossCalculator:
    def __init__(self, n_classes, weight_noobj=0.1, class_weights=None, weight_pos=1):
        self.xp = numpy
        if class_weights:
            self.class_weights = self.xp.array(class_weights, dtype=self.xp.float32)
        else:
            self.class_weights = self.xp.ones(n_classes, dtype=self.xp.float32)
        self.obj_weights = self.xp.array((weight_noobj, 1.0), dtype=self.xp.float32)
        self.weight_pos = weight_pos
        self.n_classes = n_classes

    def to_gpu(self):
        import cupy
        self.xp = cupy
        self.class_weights_cpu = self.class_weights
        self.class_weights = chainer.cuda.to_gpu(self.class_weights_cpu)
        self.obj_weights_cpu = self.obj_weights
        self.obj_weights = chainer.cuda.to_gpu(self.obj_weights)

    def to_cpu(self):
        self.xp = numpy
        self.class_weights = self.class_weights_cpu
        self.obj_weights = self.obj_weights_cpu

    def loss(self, pred, actual):
        batch_size, n_boxes, _ = actual.shape
        actual_obj_ids = self.xp.array(actual[:,:,4], dtype=self.xp.int32).reshape(batch_size * n_boxes)
        predicted_objs = pred[:,:,4:].reshape(batch_size * n_boxes, self.n_classes)
        cl_loss = F.softmax_cross_entropy(predicted_objs, actual_obj_ids, class_weight=self.class_weights)
        cl_acc = F.accuracy(predicted_objs, actual_obj_ids)

        obj_idx = self.xp.where(actual[:,:,4] > 0)
        if obj_idx[0].size > 0:
            # 教師データ側にオブジェクトが存在するgridのみ、bboxの位置を評価してlossに含める
            pred_boxes = pred[obj_idx][:,:4]
            actual_boxes = actual[obj_idx][:,:4]
            pos_loss = F.mean_squared_error(pred_boxes, actual_boxes)
            #pos_loss = F.absolute_error(pred_boxes, actual_boxes)

        else:
            pos_loss = 0

        loss = cl_loss + self.weight_pos * pos_loss
        result = {
            'loss': loss,
            'pos_loss': pos_loss,  # loss of position
            'cl_loss': cl_loss,  # loss of classification
            'cl_acc': cl_acc  # accuracy of classification
        }
        return result
