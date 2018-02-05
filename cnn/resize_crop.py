# coding: utf-8

import os
from PIL import Image

OUT_SIZE = 299
#OUT_SIZE = 224
#OUT_SIZE = 100


def main():
    for f in os.listdir('jpg'):
        img = Image.open('jpg/' + f)
        w, h = img.size
        # 縦横の長さが違う場合、短辺の長さまで長辺の両端を切り落とす
        if w > h:
            padding = int((w - h) / 2)
            img = img.crop((padding, 0, h + padding, h))
        elif w < h:
            padding = int((h - w) / 2)
            img = img.crop((0, padding, w, w + padding))
        # リサイズして保存
        img.thumbnail((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        img.save('dataset{}/'.format(OUT_SIZE) + f)


if __name__ == '__main__':
    main()
