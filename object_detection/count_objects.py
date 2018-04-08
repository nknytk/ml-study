# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree

"""
name2id = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}
id2name = {v: k for k, v in name2id.items()}
"""
name2id = {
    "aeroplane": 0,
    "bicycle": 0,
    "bird": 1,
    "boat": 0,
    "bottle": 2,
    "bus": 0,
    "car": 0,
    "cat": 1,
    "chair": 2,
    "cow": 1,
    "diningtable": 2,
    "dog": 1,
    "horse": 1,
    "motorbike": 0,
    "person": 3,
    "pottedplant": 2,
    "sheep": 1,
    "sofa": 2,
    "train": 0,
    "tvmonitor": 2
}
id2name = {
    0: "vehricle",
    1: "animal",
    2: "furniture",
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
    tx, ty, vx, vy = load_pascal_voc_dataset(sys.argv[1])

    img_cnt = {}
    obj_cnt = {}
    for objs in ty:
        _ids = set()
        for obj in objs:
            _id = obj[4]
            obj_cnt[_id] = obj_cnt.get(_id, 0) + 1
            _ids.add(_id)
        for _id in _ids:
            img_cnt[_id] = img_cnt.get(_id, 0) + 1

    for _id in sorted(id2name.keys()):
        print('{}\t{}\t{}'.format(id2name[_id], img_cnt[_id], obj_cnt[_id]))
