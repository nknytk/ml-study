# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree
from chainer.datasets import TupleDataset
from chainer.dataset.dataset_mixin import DatasetMixin

sys.path.append(os.path.dirname(__file__))
from image_util import image_to_yolodata


class ObjectDetectionDataset(DatasetMixin):
    def __init__(self, x, y, target_size=224, n_classes=1, augment=False):
        self._datasets = (x, y)
        self._length = len(x)

        self.target_size = target_size
        self.n_classes = n_classes
        self.augment = augment

    def get_example(self, i):
        image_path = self._datasets[0][i]
        objs = self._datasets[1][i]

        np_img, bboxes = image_to_yolodata(image_path, self.target_size, self.n_classes, objs, n_boxes=5, augment=self.augment)
        return np_img, bboxes

    def __len__(self):
        return self._length


def parse_position(xml_file_path, obj_name, max_obj=5):
    positions = []
    xml_root = ElementTree.parse(xml_file_path).getroot()
    for bndbox in xml_root.findall("./object[name='{}']/bndbox".format(obj_name)):
        positions.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text),
            1
        ])

    center_w = int(xml_root.find('./size/width').text) / 2
    center_h = int(xml_root.find('./size/height').text) / 2

    return positions


def pascal_voc_2012(root_dir, obj_name, limit=None):
    annotaion_dir = os.path.join(root_dir, 'Annotations')
    image_dir = os.path.join(root_dir, 'JPEGImages')

    train_x = []
    train_y = []
    val_x = []
    val_y = []

    with open(os.path.join(root_dir, 'ImageSets/Main/{}_train.txt'.format(obj_name))) as fp:
        for i, line in enumerate(fp):
            if limit and i > limit:
                break
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            train_x.append(os.path.join(image_dir, image_id + '.jpg'))
            boxes = parse_position(os.path.join(annotaion_dir, image_id + '.xml'), obj_name) if is_obj == '1' else []
            train_y.append(boxes)

    with open(os.path.join(root_dir, 'ImageSets/Main/{}_val.txt'.format(obj_name))) as fp:
        for i, line in enumerate(fp):
            if limit and i > limit:
                break
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            val_x.append(os.path.join(image_dir, image_id + '.jpg'))
            boxes = parse_position(os.path.join(annotaion_dir, image_id + '.xml'), obj_name) if is_obj == '1' else []
            val_y.append(boxes)

    train_dataset = ObjectDetectionDataset(train_x, train_y, n_classes=1, target_size=100, augment=True)
    val_dataset = ObjectDetectionDataset(val_x, val_y, n_classes=1, target_size=100, augment=False)
    return train_dataset, val_dataset
