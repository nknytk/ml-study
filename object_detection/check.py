# coding: utf-8

import random
import os
import re
import sys
from time import time
from xml.etree import ElementTree
import chainer
from chainer.functions import softmax
import numpy
from PIL import Image, ImageDraw, ImageFont

from mobile_yolo import MobileYOLO
from inception_resnet_yolo import IRYOLO
from image_util import crop_resize
from yolo_dataset import GridRetriever
from pascal_voc_loader import name2id, id2name


chainer.config.train = False
model = MobileYOLO(n_classes=2, n_base_units=32)
chainer.serializers.load_npz('results/mobile_32/model_last.npz', model)
#model = IRYOLO(n_classes=2, base_filter_num=8, ablocks=1, bblocks=2, cblocks=1)
#chainer.serializers.load_npz('results/ir_8/best_loss.npz', model)
grid_retriever = GridRetriever(model.img_size, model.img_size, model.n_grid, model.n_grid)
font = ImageFont.truetype('arial.ttf', 25)


s = time()
with open('../VOCdevkit/VOC2012/ImageSets/Main/cat_val.txt') as fp:
#with open('../VOCdevkit/VOC2012/ImageSets/Main/cat_train.txt') as fp:
    for i, line in enumerate(fp):
        image_name, _ = re.sub('\s+', ' ', line.strip()).split(' ')
        img = Image.open('../VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(image_name))
        resized_image = img.resize((model.img_size, model.img_size), Image.LANCZOS)
        w_ratio = img.size[0] / model.img_size
        h_ratio = img.size[1] / model.img_size

        np_img = numpy.asarray(resized_image.convert('RGB'), dtype=numpy.float32).transpose(2, 0, 1)
        x = chainer.Variable(numpy.array([np_img], dtype=numpy.float32))

        draw = ImageDraw.Draw(img)
        pred = model.predict(x).data[0]
        for grid_index, vec in enumerate(pred):
            obj_probas = softmax(numpy.array([vec[4:]], dtype=numpy.float32)).data[0]
            obj_id = numpy.argmax(obj_probas)
            if obj_id == 0:
                continue
            proba = obj_probas[obj_id]
            obj_name = id2name[obj_id]
            x_min, y_min, x_max, y_max = grid_retriever.restore_box(grid_index, *vec[:4])
            x_min = int(round(x_min * w_ratio))
            y_min = int(round(y_min * h_ratio))
            x_max = int(round(x_max * w_ratio))
            y_max = int(round(y_max * h_ratio))
            draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 0, 0, 1))
            draw.text((x_min, y_min), '{:.01f}'.format(proba * 100), fill=(255, 0, 0, 1), font=font)

        del(draw)

        img.save('samples/{}.jpg'.format(image_name))
        if i == 500:
            break

total_time = time() - s
print(i, total_time, total_time/i)
