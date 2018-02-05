# coding: utf-8

import os
from random import choice, random
import numpy
import chainer
from chainer.training import StandardUpdater, Trainer, extensions
from chainer.iterators import SerialIterator
from chainer.datasets import LabeledImageDataset
from chainer.optimizers import Adam
from PIL import Image, ImageOps

from vggnetbn import VGGNetBN
from googlenet import GoogLeNetBN
from resnet import ResNet50
from inception_v4 import InceptionV4
from inception_resnet_v2 import InceptionResNetV2
from fc100 import FaceClassifier100x100V, FaceClassifier100x100V2
from squeezenet import SqueezeNet
from mobilenet import MobileNet

N_EPOCHS = 200
BATCH_SIZE = 10
DEVICE = 0
#DATA_DIR = 'dataset299/'
#DATA_DIR = 'dataset224/'
DATA_DIR = 'dataset100/'


class LabeledImageDatasetWithAugmentation(LabeledImageDataset):

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.aug_functions = [
            lambda pil_img: pil_img,
            lambda pil_img: ImageOps.mirror(pil_img),
            self._random_crop,
            lambda pil_img: self._random_crop(ImageOps.mirror(pil_img))
        ]
        self.max_crop_offset = 0.05

    def _read_image_as_array(self, path, dtype):
        aug_func = choice(self.aug_functions)
        f = Image.open(path)
        try:
            image = numpy.asarray(aug_func(f), dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()
        return image

    def _random_crop(self, pil_img):
        orig_x, orig_y = pil_img.size
        tmp_x = int(orig_x * (1 + self.max_crop_offset))
        tmp_y = int(orig_y * (1 + self.max_crop_offset))
        tmp_img = pil_img.resize((tmp_x, tmp_y), Image.LANCZOS)
        offset_x = int((tmp_x - orig_x) * random())
        offset_y = int((tmp_y - orig_y) * random())
        cropped_img = tmp_img.crop((offset_x, offset_y, orig_x + offset_x, orig_y + offset_y))
        return cropped_img

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = self._read_image_as_array(full_path, self._dtype)

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label


def main():
    # input_size: 299
    #model = InceptionV4(dim_out=17)
    #model = InceptionV4(dim_out=17, base_filter_num=6, ablocks=2, bblocks=1, cblocks=1)
    #model = InceptionResNetV2(dim_out=17)
    #model = InceptionResNetV2(dim_out=17, base_filter_num=8, ablocks=1, bblocks=2, cblocks=1)

    # input_size: 224
    #model = VGGNetBN(17)  # VGGNet original size
    #model = VGGNetBN(17, 16)  # VGGNet 1/4 of filter num
    #model = GoogLeNetBN(17)  # GoogLeNet original size
    #model = GoogLeNetBN(17, 16)  # GoogleNet 1/2 filter num
    #model = GoogLeNetBN(17, 8)  # GoogleNet 1/4 filter num
    #model = ResNet50(17)  # ResNet50 original size
    #model = ResNet50(17, 32)  # ResNet50 1/2 size
    #model = ResNet50(17, 16)  # ResNet50 1/4 size
    #model = SqueezeNet(17)  #SqueezeNet original size
    #model = SqueezeNet(17, 8)  #SqueezeNet 1/2 filter num
    #model = MobileNet(17)  # MobileNet original size
    #model = MobileNet(17, 16)  # MobileNet 1/2 filter num
    #model = MobileNet(17, 8)  # MobileNet 1/4 filter num

    # input_size: 100
    #model = FaceClassifier100x100V2(n_classes=17)
    model = FaceClassifier100x100V(n_classes=17)

    optimizer = Adam()
    optimizer.setup(model)

    train_dataset = load_dataset('train.tsv', True)
    test_dataset = load_dataset('test.tsv')

    train_iter = SerialIterator(train_dataset, batch_size=BATCH_SIZE)
    test_iter = SerialIterator(test_dataset, batch_size=BATCH_SIZE, shuffle=False, repeat=False)
    updater = StandardUpdater(train_iter, optimizer, device=DEVICE)
    trainer = Trainer(updater, (N_EPOCHS, 'epoch'), out='result')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.Evaluator(test_iter, model, device=DEVICE))
    trainer.extend(extensions.PrintReport(['main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    #trainer.extend(extensions.snapshot_object(model, 'snapshot_{.updater.epoch}.model'))

    trainer.run()

    chainer.serializers.save_npz('result/model.npz', model.to_cpu())


def load_dataset(tsv, aug=False):
    pairs = []
    with open(tsv) as fp:
        for line in fp:
            label, image_file = line.strip().split('\t')
            pairs.append((DATA_DIR + image_file, numpy.int32(label)))
    if aug:
        return LabeledImageDatasetWithAugmentation(pairs)
    else:
        return LabeledImageDataset(pairs)


if __name__ == '__main__':
    main()
