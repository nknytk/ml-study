# coding: utf-8

from random import random, choice
import numpy
from PIL import Image, ImageOps


def image_to_yolodata(img_path, target_size, n_classes, boxes, n_boxes=5, augment=False):
    """
    Input:
      target_size: resize後の1辺の画素数
      n_classes: 検出すべきオブジェクトの種類数(no objectを除く)
      boxes: オブジェクトbounding boxの情報を含んだ配列。
        各オブジェクトは(xmin, ymin, xmax, ymax, class_id)の配列
    Output:
      (3, target_size, target_size)の画像, bboxの配列
      bboxの配列は長さ n_tiles**2 で、各要素は
      (w_min, h_min, w_max, h_max, class_id)
    """
    orig_img = Image.open(img_path)
    w, h = orig_img.size

    if augment:
        #t_img, t_boxes = orig_img, boxes if random() > 0.5 else horizontal_flip(orig_img, boxes)
        if random() > 0.5:
            t_img, t_boxes = orig_img, boxes
        else:
            t_img, t_boxes = horizontal_flip(orig_img, boxes)
        max_noise = min(w, h) * 0.2
        w_noise = int(max_noise * (random() - 1))
        h_noise = int(max_noise * (random() - 1))

        t_img, t_boxes = resize(t_img, t_boxes, target_size, w_noise, h_noise)
    else:
        t_img, t_boxes = resize(orig_img, boxes, target_size)

    np_img = numpy.asarray(t_img.convert('RGB'), dtype=numpy.float32).transpose(2, 0, 1)

    img_center = target_size / 2
    t_boxes.sort(key=lambda x: sum([(x[i] - img_center)**2 for i in range(4)]))  # 端の4点が中心から近い順にソートする
    np_boxes = numpy.zeros((n_boxes, 5), dtype=numpy.float32)
    for i, box in enumerate(t_boxes[:n_boxes]):
        w_min, h_min, w_max, h_max, cls_id = box
        np_boxes[i][0] = w_min / target_size
        np_boxes[i][1] = h_min / target_size
        np_boxes[i][2] = w_max / target_size
        np_boxes[i][3] = h_max / target_size
        np_boxes[i][4] = cls_id

    return np_img, np_boxes


def horizontal_flip(img, boxes):
    """ 画像とbounding boxの座標の両方を左右反転する """
    flipped_img = ImageOps.mirror(img)
    X, Y = img.size
    flipped_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax, cls_id = box
        flipped_xmin = X - xmax
        flipped_xmax = X - xmin
        flipped_boxes.append((flipped_xmin, ymin, flipped_xmax, ymax, cls_id))
    return flipped_img, flipped_boxes


def resize(img, boxes, size, w_noise=0, h_noise=0):
    w, h = img.size
    w2 = w - abs(w_noise)
    h2 = h - abs(h_noise)
    w_ratio = size / w2
    h_ratio = size / h2

    xmin = max(0, w_noise)
    ymin = max(0, h_noise)
    xmax = min(w, w + w_noise)
    ymax = min(h, h + w_noise)

    cropped_img = img.crop((xmin, ymin, xmax, ymax)).reshape((size, size), Image.LANCZOS)
    offset_boxes = []
    for box in boxes:
        bxmin = max(0, (box[0] - xmin) * w_ratio)
        bymin = max(0, (box[1] - ymin) * h_ratio)
        bxmax = min(size, (box[2] - xmin) * w_ratio)
        bymax = min(size, (box[3] - ymin) * h_ratio)
        if bxmin < size and bymin < size and bxmax > 0 and bymax > 0:
            offset_boxes.append((bxmin, bymin, bxmax, bymax, box[4]))

    return cropped_img, offset_boxes


def crop_resize(img, boxes, size, w_noise=0, h_noise=0):
    """
    縦横の長さが違う場合、短辺の長さまで長辺の両端を切り落とし、中央の正方形を切り抜く。
    [wh]_noizeが指定されている場合、対応する画素数分切り抜き箇所をずらしてノイズを加える。
    切り抜き範囲とbounding boxが干渉する場合、boxが半分以上残るなら
    切り抜かれたboxをbounding boxとする。
    """
    w, h = img.size
    if w > h:
        if h_noise > 0:
            h_min = h_noise
            h_max = h
        else:
            h_min = 0
            h_max = h + h_noise
        crop_size = h_max - h_min

        w_offset = int((w - h) / 2)
        w_min = w_offset + w_noise if w_offset > abs(w_noise) else w_offset
        w_max = w_min + crop_size

    elif w < h:
        if w_noise > 0:
            w_min = w_noise
            w_max = w
        else:
            w_min = 0
            w_max = w + w_noise
        crop_size = w_max - w_min

        h_offset = int((h - w) / 2)
        h_min = h_offset + h_noise if h_offset > abs(h_noise) else h_offset
        h_max = h_min + crop_size

    else:
        if w_noise > 0:
            w_min = w_noise
            w_max = w
        else:
            w_min = 0
            w_max = w + w_noise
        crop_size = w_max - w_min

        h_min, h_max = choice([(0, crop_size), (h - crop_size, h)])

    cropped_img = img.crop((w_min, h_min, w_max, h_max))
    resized_img = cropped_img.resize((size, size), Image.LANCZOS)
    resize_ratio = size / crop_size

    cropped_boxes = []
    for box in boxes:
        bw_min, bh_min, bw_max, bh_max, cls_id = box
        # bboxが切り抜き範囲から完全に外れていたら、返却するbboxから除外する
        if bw_max <= w_min or w_max <= bw_min or bh_max <= h_min or h_max <= bh_min:
            continue
        # bboxが切り抜き範囲に完全に含まれていたら、offsetだけ調整して返却する
        if w_min <= bw_min and bw_max <= w_max and h_min <= bh_min and bh_max <= h_max:
            cropped_boxes.append((
                int((bw_min - w_min) * resize_ratio),
                int((bh_min - h_min) * resize_ratio),
                int((bw_max - w_min) * resize_ratio),
                int((bh_max - h_min) * resize_ratio),
                cls_id
            ))
            continue
        # bboxと切り抜き範囲が一部だけ重なっていたら、重なり部分をbboxとし、面積が半分以上残っているならbboxとして返す
        i_box = (max(w_min, bw_min), max(h_min, bh_min), min(w_max, bw_max), min(h_max, bh_max))
        i_area = (i_box[2] - i_box[0]) * (i_box[3] - i_box[1])
        o_area = (box[2] - box[0]) * (box[3] - box[1])
        if i_area/o_area > 0.5:
            cropped_boxes.append((
                int((i_box[0] - w_min) * resize_ratio),
                int((i_box[1] - h_min) * resize_ratio),
                int((i_box[2] - w_min) * resize_ratio),
                int((i_box[3] - h_min) * resize_ratio),
                cls_id,
            ))

    return resized_img, cropped_boxes
