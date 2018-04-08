# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree


name2id = {
    "no_obj": 0,
    "head": 1,
}
id2name = {v: k for k, v in name2id.items()}


def object_positions(xml_file_path):
    positions = []
    xml_root = ElementTree.parse(xml_file_path).getroot()
    for obj in xml_root.findall('./object'):
        for part in obj.findall('part'):
            part_name = part.find('name').text
            if part_name not in name2id:
                continue
            bndbox = part.find('bndbox')
            if not bndbox:
                continue

            positions.append([
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text),
                name2id[part_name]
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
        l = 0
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            if is_obj != '1':
                if l >= 200:
                    continue
                else:
                    l += 1
            train_x.append(os.path.join(image_2007_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2007_dir, image_id + '.xml'))
            train_y.append(obj_pos)
    with open(os.path.join(root_dir, 'VOC2012/ImageSets/Main/person_train.txt')) as fp:
        l = 0
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            if is_obj != '1':
                if l >= 200:
                    continue
                else:
                    l += 1
            train_x.append(os.path.join(image_2012_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2012_dir, image_id + '.xml'))
            train_y.append(obj_pos)
    with open(os.path.join(root_dir, 'VOC2012/ImageSets/Main/person_val.txt')) as fp:
        l = 0
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            if is_obj != '1':
                if l >= 200:
                    continue
                else:
                    l += 1
            val_x.append(os.path.join(image_2012_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_2012_dir, image_id + '.xml'))
            val_y.append(obj_pos)

    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    res = load_pascal_voc_dataset(sys.argv[1])
    print(len(res[1]), sum([len(l) for l in res[1] if l]))
