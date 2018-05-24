# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree


name2id = {
    "aeroplane": 1,
    "bicycle": 1,
    "bird": 2,
    "boat": 1,
    "bottle": 0,
    "bus": 1,
    "car": 1,
    "cat": 2,
    "chair": 3,
    "cow": 2,
    "diningtable": 0,
    "dog": 2,
    "horse": 2,
    "motorbike": 1,
    "person": 3,
    "pottedplant": 0,
    "sheep": 2,
    "sofa": 0,
    "train": 1,
    "tvmonitor": 0
}
id2name = {
    0: "no_obj",
    1: "vehicle",
    2: "animal",
    3: "person"
}


def object_positions(xml_file_path):
    positions = []
    xml_root = ElementTree.parse(xml_file_path).getroot()
    for obj in xml_root.findall('./object'):
        bndbox = obj.find('bndbox')
        if not bndbox:
            continue
        obj_name = obj.find('name').text

        positions.append([
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymax').text),
            name2id[obj_name]
        ])

    return positions


def load_pascal_voc_dataset(root_dir):
    """
    Pascal Voc Devkitから訓練・評価データを取得する。
    2007の全データと2012のtrainデータを訓練データ、2012のvalデータを評価データとする。
    """
    image_2012_dir = os.path.join(root_dir, 'VOC2012/JPEGImages')
    image_2007_dir = os.path.join(root_dir, 'VOC2007/JPEGImages')
    annotation_2012_dir = os.path.join(root_dir, 'VOC2012/Annotations')
    annotation_2007_dir = os.path.join(root_dir, 'VOC2007/Annotations')

    train_x, train_y, val_x, val_y = [], [], [], []
    with open(os.path.join(root_dir, 'VOC2007/ImageSets/Main/person_trainval.txt')) as fp:
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            train_x.append(os.path.join(image_2007_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2007_dir, image_id + '.xml'))
            train_y.append(obj_pos)
    with open(os.path.join(root_dir, 'VOC2012/ImageSets/Main/person_train.txt')) as fp:
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            train_x.append(os.path.join(image_2012_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2012_dir, image_id + '.xml'))
            train_y.append(obj_pos)
    with open(os.path.join(root_dir, 'VOC2012/ImageSets/Main/person_val.txt')) as fp:
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            val_x.append(os.path.join(image_2012_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2012_dir, image_id + '.xml'))
            val_y.append(obj_pos)

    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    load_pascal_voc_dataset(sys.argv[1])
