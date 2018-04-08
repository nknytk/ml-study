# coding: utf-8

import os
import sys
from random import random, choice
import numpy
from PIL import Image, ImageOps
from chainer.dataset.dataset_mixin import DatasetMixin

sys.path.append(os.path.dirname(__file__))
from image_util import horizontal_flip, crop_resize


class YoloDataset(DatasetMixin):
    def __init__(self, x, y, target_size=224, n_grid=7, augment=False):
        self._datasets = (x, y)
        self._length = len(x)

        self.target_size = target_size
        self.n_grid = n_grid
        self.augment = augment
        self.grid_retriever = GridRetriever(target_size, target_size, n_grid, n_grid)

        max_class_id = 0
        for objs in y:
            for obj in objs:
                max_class_id = max(max_class_id, obj[4])
        self.n_classes = max_class_id + 1

    def __len__(self):
        return self._length

    def get_example(self, i):
        image_path = self._datasets[0][i]
        objs = self._datasets[1][i]
        np_img, bboxes = self.convert(image_path, objs)
        return np_img, bboxes

    def convert(self, img_path, objs):
        """
        Input:
          img_path: 画像ファイルのパス
          objs: オブジェクトbounding boxの情報を含んだ配列。各オブジェクトは(xmin, ymin, xmax, ymax, class_id)
        Output:
          (3, target_size, target_size)の画像, bboxの配列
          bboxの配列は長さ n_grid**2 で、各要素は
          (中心点のgrid内x, 中心点のgrid内y, 幅, 高さ, class_id)
        """
        orig_img = Image.open(img_path)
        w, h = orig_img.size

        if self.augment:
            if random() > 0.5:
                t_img, t_boxes = orig_img, objs
            else:
                t_img, t_boxes = horizontal_flip(orig_img, objs)
            max_noise = min(w, h) * 0.2
            w_noise = int(max_noise * (random() - 0.5))
            h_noise = int(max_noise * (random() - 0.5))

            t_img, t_boxes = crop_resize(t_img, t_boxes, self.target_size, w_noise, h_noise)
        else:
            t_img, t_boxes = crop_resize(orig_img, objs, self.target_size)

        np_img = numpy.asarray(t_img.convert('RGB'), dtype=numpy.float32).transpose(2, 0, 1)

        np_boxes = numpy.zeros((self.n_grid**2, 5 + self.n_classes), dtype=numpy.float32)
        for box in t_boxes:
            w_min, h_min, w_max, h_max, cls_id = box
            grid_idx, grid_center_x, grid_center_y, relative_w, relative_h = self.grid_retriever.grid_position(w_min, h_min, w_max, h_max)
            np_boxes[grid_idx][0] = grid_center_x
            np_boxes[grid_idx][1] = grid_center_y
            np_boxes[grid_idx][2] = relative_w
            np_boxes[grid_idx][3] = relative_h
            np_boxes[grid_idx][4] = cls_id

        return np_img, np_boxes


class GridRetriever:
    def __init__(self, img_x, img_y, n_grid_x, n_grid_y):
        self.img_x = img_x
        self.img_y = img_y
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.grid_size_x = int(img_x/n_grid_x)
        self.grid_size_y = int(img_x/n_grid_y)

    def grid_position(self, x_min, y_min, x_max, y_max):
        """
        Input:
          オブジェクト位置座標 (x_min. y_min, x_max, y_max)
        Output:
          grid index, grid内での中心点座標位置x, y, オブジェクト大きさw, h
        """
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_grid_index = int(x_center/self.grid_size_x)
        y_grid_index = int(y_center/self.grid_size_y)
        grid_index = y_grid_index * self.n_grid_x + x_grid_index
        x_grid_center = (x_center - x_grid_index * self.grid_size_x)/ self.grid_size_x
        y_grid_center = (y_center - y_grid_index * self.grid_size_y)/ self.grid_size_y
        x_size_relative = (x_center - x_min) / self.img_x
        y_size_relative = (y_center - y_min) / self.img_y
        return grid_index, x_grid_center, y_grid_center, x_size_relative, y_size_relative


    def restore_box(self, grid_index, x_grid_center, y_grid_center, x_size, y_size):
        """
        Input:
          grid index, grid内での中心点座標位置x, y, オブジェクト大きさw, h
        Output:
          オブジェクト位置座標 (x_min. y_min, x_max, y_max)
        """
        y_grid_index = int(grid_index / self.n_grid_x)
        x_grid_index = grid_index % self.n_grid_y
        x_center = int((x_grid_index + x_grid_center) * self.grid_size_x)
        y_center = int((y_grid_index + y_grid_center) * self.grid_size_y)
        x_min = x_center - int(x_size * self.img_x)
        y_min = y_center - int(y_size * self.img_y)
        x_max = x_center + int(x_size * self.img_x)
        y_max = y_center + int(y_size * self.img_y)
        return x_min, y_min, x_max, y_max
