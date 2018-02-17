# coding: utf-8

import os
import sys
from random import choice, random
import numpy
import chainer
from chainer.training import StandardUpdater, Trainer, extensions
from chainer.iterators import SerialIterator
from chainer.datasets import LabeledImageDataset
from chainer.optimizers import Adam

sys.path.append(os.path.dirname(__file__))
from dataset import ObjectDetectionDataset, pascal_voc_2012
from nano_yolo import NanoYOLO, MicroYOLO

DEVICE = -1
BATCH_SIZE = 10
N_EPOCHS = 100
DATASET_ROOT = ''

def main():
    model = NanoYOLO(n_classes=1)
    #model = MicroYOLO(n_classes=1)

    optimizer = Adam()
    optimizer.setup(model)

    train_dataset, test_dataset =  pascal_voc_2012(DATASET_ROOT, 'person', 100)

    train_iter = SerialIterator(train_dataset, batch_size=BATCH_SIZE)
    test_iter = SerialIterator(test_dataset, batch_size=BATCH_SIZE, shuffle=False, repeat=False)
    updater = StandardUpdater(train_iter, optimizer, device=DEVICE)
    trainer = Trainer(updater, (N_EPOCHS, 'epoch'), out='result')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.Evaluator(test_iter, model, device=DEVICE))
    trainer.extend(extensions.PrintReport([
        'main/loss', 'validation/main/loss',
        'main/cl_acc', 'validation/main/cl_acc',
        'main/pos_loss', 'validation/main/pos_loss',
    ]))

    trainer.run()

    chainer.serializers.save_npz('result/model.npz', model.to_cpu())


if __name__ == '__main__':
    main()
