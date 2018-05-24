# coding: utf-8

import os
import sys
from random import choice, random
import numpy
import chainer
from chainer.training import StandardUpdater, Trainer, extensions, triggers
from chainer.iterators import SerialIterator
from chainer.datasets import LabeledImageDataset
from chainer.optimizers import Adam

sys.path.append(os.path.dirname(__file__))
from load_3obj import load_pascal_voc_dataset
from yolo_dataset import YoloDataset
from mobile_yolo import MobileYOLO

DEVICE = 0
BATCH_SIZE = 20
N_EPOCHS = 250
DATASET_ROOT = '../VOCdevkit'
RESULT_DIR = 'results/mobile_6.20'
model_class = MobileYOLO


def main():
    train_x, train_y, val_x, val_y = load_pascal_voc_dataset(DATASET_ROOT)
    train_dataset = YoloDataset(train_x, train_y, target_size=model_class.img_size, n_grid=model_class.n_grid, augment=True)
    test_dataset = YoloDataset(val_x, val_y, target_size=model_class.img_size, n_grid=model_class.n_grid, augment=False)

    class_weights = [1.0 for i in range(train_dataset.n_classes)]
    class_weights[0] = 0.2
    model = model_class(n_classes=train_dataset.n_classes, n_base_units=6, class_weights=class_weights)
    if os.path.exists(RESULT_DIR + '/model_last.npz'):
        print('continue from previous result')
        chainer.serializers.load_npz(RESULT_DIR + '/model_last.npz', model)
    optimizer = Adam()
    optimizer.setup(model)


    train_iter = SerialIterator(train_dataset, batch_size=BATCH_SIZE)
    test_iter = SerialIterator(test_dataset, batch_size=BATCH_SIZE, shuffle=False, repeat=False)
    updater = StandardUpdater(train_iter, optimizer, device=DEVICE)
    trainer = Trainer(updater, (N_EPOCHS, 'epoch'), out=RESULT_DIR)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.Evaluator(test_iter, model, device=DEVICE))
    trainer.extend(extensions.PrintReport([
        'main/loss', 'validation/main/loss',
        'main/cl_loss', 'validation/main/cl_loss',
        'main/cl_acc', 'validation/main/cl_acc',
        'main/pos_loss', 'validation/main/pos_loss',
    ]))
    trainer.extend(extensions.snapshot_object(model, 'best_loss.npz'), trigger=triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, 'best_classification.npz'), trigger=triggers.MaxValueTrigger('validation/main/cl_acc'))
    trainer.extend(extensions.snapshot_object(model, 'best_position.npz'), trigger=triggers.MinValueTrigger('validation/main/pos_loss'))
    trainer.extend(extensions.snapshot_object(model, 'model_last.npz'), trigger=(1, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
