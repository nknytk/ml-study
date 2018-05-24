# coding: utf-8

# original implementation around qt5: https://qiita.com/odaman68000/items/c8c4093c784bff43d319

import sys
from time import sleep, time
from multiprocessing import Process, Queue
from threading import Thread
import traceback

import chainer
from chainer.functions import softmax
import numpy
import cv2
from  PyQt5 import QtCore, QtGui, QtWidgets

from mobile_yolo import MobileYOLO, MicroYOLO
from yolo_dataset import GridRetriever
from pascal_voc_loader import name2id, id2name

chainer.config.train = False
detector = MobileYOLO(n_classes=4, n_base_units=6)
chainer.serializers.load_npz('models/3obj_6.npz', detector)
grid_retriever = GridRetriever(detector.img_size, detector.img_size, detector.n_grid, detector.n_grid)


def streaming_detection():
    app = QtWidgets.QApplication(sys.argv)
    # 遅延よりFPSを優先する場合、bufferに画像を貯めて処理画像が途切れないように設定
    #window = ImageWidget(3, 1, 3)
    # bufferを減らすと、FPSとスムーズさを犠牲に遅延を減らすことができる
    window = ImageWidget(2)
    window.show()

    try:
        app.exec_()
    except:
        window.stop()


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, n_procs=3):
        super().__init__()
        self.image = None
        self.camera = cv2.VideoCapture(0)
        self.n_procs = n_procs
        self.in_q = Queue()
        self.out_q = Queue()
        self.stop_q = Queue()
        self.capture_th = Process(target=self.capture)
        self.capture_th.start()
        self.draw_th = Thread(target=self.draw)
        self.draw_th.start()
        self.detect_procs = [Process(target=detect_face, args=(self,)) for i in range(self.n_procs)]
        for proc in self.detect_procs:
            proc.start()
            sleep(1/self.n_procs)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.image is None:
            painter.setPen(QtCore.Qt.black)
            painter.setBrush(QtCore.Qt.black)
            painter.drawRect(0, 0, self.width(), self.height())
            return
        pixmap = self.create_QPixmap(self.image)
        painter.drawPixmap(0, 0, self.image.shape[1], self.image.shape[0], pixmap)

    def set_image(self, image):
        self.image = image
        self.update()

    def create_QPixmap(self, image):
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 4, QtGui.QImage.Format_ARGB32_Premultiplied)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def capture(self):
        while self.stop_q.empty():
            _, img = self.camera.read()
            if not self.in_q.empty():
                continue
            self.in_q.put((time(), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.camera.release()

    def draw(self):
        cnt = 0
        t = time()
        last_update = 0
        while self.stop_q.empty():
            try:
                timestamp, img2draw = self.out_q.get(block=True, timeout=3)
                if timestamp < last_update:
                    continue
                self.set_image(cv2.cvtColor(img2draw, cv2.COLOR_RGB2BGRA))
                last_update = timestamp

                cnt += 1
                if cnt % 30 == 0:
                    n = time()
                    print('{} FPS'.format(30 / (n - t)))
                    t = n

            except:
                continue

    def stop(self):
        self.stop_q.put(1)
        self.capture_th.join()
        self.draw_th.join()
        for proc in self.detect_procs:
            proc.join(1)

    def closeEvent(self, event):
        event.accept()
        print('stop')
        self.stop()


def detect_face(controller):
    colors = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255)
    }
    while controller.stop_q.empty():
        try:
            timestamp, image = controller.in_q.get(block=True, timeout=3)
        except:
            sleep(0.001)
            continue

        resized_image = cv2.resize(image, (detector.img_size, detector.img_size))
        w_ratio = image.shape[1] / detector.img_size
        h_ratio = image.shape[0] / detector.img_size

        x = chainer.Variable(numpy.array([resized_image.transpose(2, 0, 1)], dtype=numpy.float32))

        pred = detector.predict(x).data[0]
        for grid_index, vec in enumerate(pred):
            obj_probas = softmax(numpy.array([vec[4:]], dtype=numpy.float32)).data[0]
            obj_id = numpy.argmax(obj_probas)
            if obj_id == 0:
                continue
            proba = obj_probas[obj_id]
            if proba < 0.5 or obj_id == 3 and proba < 0.7:
                continue
            x_min, y_min, x_max, y_max = grid_retriever.restore_box(grid_index, *vec[:4])
            x_min = int(round(x_min * w_ratio))
            y_min = int(round(y_min * h_ratio))
            x_max = int(round(x_max * w_ratio))
            y_max = int(round(y_max * h_ratio))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[obj_id], thickness=2)

        controller.out_q.put((timestamp, image))


if __name__ == '__main__':
    streaming_detection()
