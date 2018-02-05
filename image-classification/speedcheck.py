# coding: utf-8

import os
from time import time
import numpy
import chainer
import cupy
from PIL import Image

from vggnetbn import VGGNetBN
from googlenet import GoogLeNetBN
from resnet import ResNet50
from inception_v4 import InceptionV4
from inception_resnet_v2 import InceptionResNetV2
from fc100 import FaceClassifier100x100V, FaceClassifier100x100V2
from squeezenet import SqueezeNet
from mobilenet import MobileNet


test_images = {}
image_names = os.listdir('dataset100')[:30]
for img_size in 100, 224, 299:
    images = []
    for f in image_names:
        pil_img = Image.open('dataset{}/{}'.format(img_size, f)).convert('RGB')
        np_img = numpy.asarray(pil_img, dtype=numpy.float32).transpose(2, 0, 1)
        images.append(np_img)
    test_images[img_size] = images


def check_speed(model, test_images_np):
    start_time = time()
    for img in test_images_np:
        x = chainer.Variable(numpy.array([img], dtype=numpy.float32))
        pred = model.predict(x)
    avg_time = (time() - start_time) / len(test_images_np)

    model.to_gpu()
    start_time = time()
    x = chainer.Variable(cupy.array(test_images_np, dtype=numpy.float32))
    pred = model.predict(x)
    batch_time = (time() - start_time) / len(test_images_np)

    return avg_time, batch_time


def main():
    test_patterns = [
        ('VGGNetBN', VGGNetBN(17), 224),
        ('VGGNetBNHalf', VGGNetBN(17, 32), 224),
        ('VGGNetBNQuater', VGGNetBN(17, 16), 224),
        ('GoogLeNetBN', GoogLeNetBN(17), 224),
        ('GoogLeNetBNHalf', GoogLeNetBN(17, 16), 224),
        ('GoogLeNetBNQuater', GoogLeNetBN(17, 8), 224),
        ('ResNet50', ResNet50(17), 224),
        ('ResNet50Half', ResNet50(17, 32), 224),
        ('ResNet50Quater', ResNet50(17, 16), 224),
        ('SqueezeNet', SqueezeNet(17), 224),
        ('SqueezeNetHalf', SqueezeNet(17, 8), 224),
        ('MobileNet', MobileNet(17), 224),
        ('MobileNetHalf', MobileNet(17, 16), 224),
        ('MobileNetQuater', MobileNet(17, 8), 224),
        ('InceptionV4', InceptionV4(dim_out=17), 299),
        ('InceptionV4S', InceptionV4(dim_out=17, base_filter_num=6, ablocks=2, bblocks=1, cblocks=1), 299),
        ('InceptionResNetV2', InceptionResNetV2(dim_out=17), 299),
        ('InceptionResNetV2S', InceptionResNetV2(dim_out=17, base_filter_num=8, ablocks=1, bblocks=2, cblocks=1), 299),
        ('FaceClassifier100x100V', FaceClassifier100x100V(17), 100),
        ('FaceClassifier100x100V2', FaceClassifier100x100V2(17), 100)
    ]

    for model_name, model, test_size in test_patterns:
        oltp_cpu, batch_gpu = check_speed(model, test_images[test_size])
        print('{}\t{:.02f}\t{:.02f}'.format(model_name, oltp_cpu * 1000, batch_gpu * 1000))


if __name__ == '__main__':
    main()
