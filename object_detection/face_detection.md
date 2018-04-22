## 顔の検知

### 準備するもの

* OS: Ubuntu または Raspbian
* USB Webカメラ
   - 筆者の機材の都合上、テストされている解像度は640x480のみ
* デスクトップ環境
   - 描画負荷軽減のため、モニタ出力解像度を1280x720とする。1920x1080にするとFPSが0.5下がる
* Python3
* python3-qt5
* chainer 3.5.0
* Pillow 5.1.0
* OpenCV 3.4.1 with Python3 binding

### 実行方法

```
$ python3 demo.py
```

### モデルについて

YOLOを参考に、MobileNetを改造したCNNモデルを使用

#### パラメータ

初期化時に`n_base_units`の値を指定し、モデルの表現力を調節できる。  
値を大きくすると、速度と引き換えに精度が上がる傾向にある。`n_base_units=32`でMobileNetと同等の負荷となる。

| n_base_units | FPS @ Raspberry Pi 3 Model B |
| --- | --- |
| 3 | 6.5 |
| 4 | 5.7 |
| 6 | 4.4 |
| 8 | 3.6 |
| 12 | 2.6 |
| 32 | 0.9 |


#### 訓練済みモデル

Pascal VOC 2012のtrainデータ、Pascal VOC 2007の全データから

* `head` を含んだ全画像
* `head` を含まない画像のうち、ランダムに選択した2%の画像

の約600件で訓練されたモデルと、訓練のログが`results/mobile_${n_base_units}`以下に保存されている。
