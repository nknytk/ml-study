# coding: utf-8

import numpy
import chainer
from chainer.training import StandardUpdater, Trainer, extensions
from chainer.iterators import SerialIterator
from chainer.datasets import LabeledImageDataset
from chainer.optimizers import Adam

from vggnetbn import VGGNetBN
from googlenetbn import GoogLeNetBN
from resnet import ResNet50
from inception_v4 import InceptionV4
from inception_resnet_v2 import InceptionResNetV2
from fc100 import FaceClassifier100x100V, FaceClassifier100x100V2

N_EPOCHS = 100
BATCH_SIZE = 10
DEVICE = 0
#DATA_DIR = 'dataset299/'
DATA_DIR = 'dataset224/'
#DATA_DIR = 'dataset100/'


def main():
    #model = InceptionV4(dim_out=17)
    #model = InceptionV4(dim_out=17, base_filter_num=6, ablocks=2, bblocks=1, cblocks=1)
    #model = InceptionResNetV2(dim_out=17)
    #model = InceptionResNetV2(dim_out=17, base_filter_num=8, ablocks=1, bblocks=2, cblocks=1)
    #model = FaceClassifier100x100V2(n_classes=17)
    #model = FaceClassifier100x100V(n_classes=17)
    #model = VGGNetBN(17)
    #model = VGGNetBN(17, 16)
    model = GoogLeNetBN()
    #model = ResNet50()
    optimizer = Adam()
    optimizer.setup(model)

    train_dataset = load_dataset('train.tsv')
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


def load_dataset(tsv):
    pairs = []
    with open(tsv) as fp:
        for line in fp:
            label, image_file = line.strip().split('\t')
            pairs.append((DATA_DIR + image_file, numpy.int32(label)))
    return LabeledImageDataset(pairs)


if __name__ == '__main__':
    main()
