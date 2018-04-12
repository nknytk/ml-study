# 性能検証レポート

## 目的

今回実装したinception-V4とinception-resnet-V2の精度とコストを確かめる。気になるのは以下の3点  

* 実装が間違っていて精度が出ない、ということがないか
* その他の有名どころのモデルと比べて精度が高いのか
* ImageNetのような大規模なデータセットではなく、個人が扱えるような小規模なデータセットでも有効なものか

## 評価方法

### データセット

[17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)の画像を、各カテゴリ80枚中70枚を学習用、10枚を評価用に分けて使用する。  
オリジナル画像のサイズは揃っていないため、短辺に合わせて長辺の両端を切り落とし、CNNの入力用にリサイズする。また、学習時には訓練画像にランダムに以下のフィルタを適用し、汎化性能の向上を図る。

* 無加工
* 左右反転
* ランダムクロップ
* 左右反転とランダムクロップ

### 評価環境

| 項目 | 値 |
| ---- | --- |
| CPU  | Ryzen 1700 (8C16T) |
| GPU  | GeForce GTX 1070 |
| Memory | 16GB |
| Disk | M.2 SSD 256GB |
| OS | Ubuntu 16.04 |
| Cuda | 9.0 |
| Python | 3.5.2 |
| chainer | 3.2.0 |
| cupy | 2.1.0 |

### 評価軸

200epoch学習させ、精度と計算コストを確認する。

| 項目 | 説明 |
| ---- | ---- |
| 精度 | 評価データに対する正解率の最高値(%) |
| 学習時間 | GPUを使用し、`batchsize=10` で200epochの学習と評価を行うのにかかった時間(s) |
| CPU推論時間 | 環境変数`OMP_NUM_THREADS=1`を設定し、CPUを1Coreのみ使用して1枚ずつ推論を行うのにかかった時間の平均(image/ms) |
| GPUバッチ推論時間 | GPUを使用し、batchsize=30で推論を行った際の画像1枚あたりの所要時間(image/ms) |
| 推論時間@RPi | Raspberry Pi 3 Model Bで、環境変数`OMP_NUM_THREADS=1`を設定し、  
CPUを1Coreのみ使用して1枚ずつ推論を行うのにかかった時間の平均(image/ms) |
| モデルサイズ | 学習済モデルをserializeして保存した時のnpzファイルサイズ(MB) |

### モデル

入力画像サイズ`224x224`のモデル

| モデル | 説明 |
| --- | --- |
| VGGNetBN | VGGNetにBatch Normalizationを追加 |
| VGGNetBNQuater | VGGNetBNのフィルタ数を半分にしたモデル |
| GoogLeNetBN | GoogLeNetにBatchNormalizationを追加 |
| GoogLeNetBNHalf | GoogLeNetBNHalfのフィルタ数を1/2にしたモデル |
| GoogLeNetBNQuater | GoogLeNetBNHalfのフィルタ数を1/2にしたモデル |
| ResNet50 | ResNet50のchainer実装 |
| ResNet50Half | ResNet50のフィルタ数を1/2にしたモデル |
| ResNet50Quater | ResNet50のフィルタ数を1/4にしたモデル |
| SqueezeNet | SqueezeNetのchainer実装 |
| SqueezeNetHalf | SqueezeNetのフィルタ数を1/2にしたモデル |
| MobileNet | MobileNetのchainer実装 |
| MobileNetHalf | MobileNetのフィルタ数を1/2にしたモデル |
| MobileNetQuater | MobileNetのフィルタ数を1/4にしたモデル |

入力画像サイズ`299x299`のモデル

| モデル | 説明 |
| --- | --- |
| InceptionV4 | 今回chainerで実装したinception-v4 |
| InceptionV4S | inception-v4のフィルタ数と深さを両方減らしたモデル |
| InceptionResNetV2 | 今回chainerで実装したinception-resnet-v2 |
| InceptionResNetV2 | inception-resnet-v2のフィルタ数と深さを両方減らしたモデル |

入力画像サイズ`100x100`のモデル

| モデル | 説明 |
| --- | --- |
| FaceClassifier100x100V | Raspberry Piでの顔分類用軽量CNN 速度重視版 |
| FaceClassifier100x100V2 | Raspberry Piでの顔分類用軽量CNN 精度重視版 |

## 結果

| モデル                 | 精度 | 学習時間  | CPU推論時間 | GPUバッチ推論時間 | 推論時間@RPi | モデルサイズ |
| ---------------------- | ---- | --------- | ----------- | ----------------- | ------------ | ------------ |
| VGGNetBN               | 11.1 | 5377      | 1260.59     | 27.57             | 実行不能     | 474M         |
| VGGNetBNQuater         | 71.7 | 1096      | 128.81      | 0.53              | 2442.67      | 30M          |
| GoogLeNetBN            | 90.5 | 4348      | 274.40      | 9.48              | 5031.91      | 53M          |
| GoogLeNetBNHalf        | 91.7 | 3445      | 118.00      | 8.39              | 1851.46      | 14M          |
| GoogLeNetBNQuater      | 90.5 | 2932      | 66.82       | 3.60              | 845.61       | 3.5M         |
| ResNet50               | 89.9 | 2510      | 405.26      | 5.78              | 8000.83      | 84M          |
| ResNet50Half           | 88.8 | 2315      | 149.55      | 2.00              | 2668.44      | 22M          |
| ResNet50Quater         | 87.0 | 2085      | 69.80       | 1.53              | 1065.56      | 5.4M         |
| SqueezeNet             | 87.0 | 1379      | 129.01      | 1.04              | 2285.15      | 2.7M         |
| SqueezeNetHalf         | 88.8 | 1348      | 59.60       | 0.62              | 951.39       | 736K         |
| MobileNet              | 91.7 | 7530      | 110.57      | 8.59              | 1939.66      | 7.9M         |
| MobileNetHalf          | 90.0 | 4268      | 52.11       | 4.26              | 818.22       | 2.1M         |
| MobileNetQuater        | 86.4 | 3007      | 29.66       | 2.24              | 407.22       | 602K         |
| InceptionV4            | 84.7 | 7873      | 1228.40     | 26.90             | 24491.11     | 147M         |
| InceptionV4S           | 91.7 | 3056      | 86.01       | 1.14              | 1278.07      | 1.9M         |
| InceptionResNetV2      | 92.9 | 7493      | 893.02      | 5.87              | 17431.35     | 105M         |
| InceptionResNetV2S     | 92.3 | 2539      | 90.24       | 1.03              | 1370.79      | 2.7M         |
| FaceClassifier100x100V | 89.9 | 362       | 14.91       | 0.18              | 231.38       | 379K         |
| FaceClassifier100x100V2| 90.5 | 411       | 29.10       | 0.17              | 500.85       | 4.1M         |

試したモデルの中ではinception-resnet-v2が最も高い精度を示した。実装は間違っていなさそうである。  
ImageNet向けのモデルをそのまま使用した際は学習・推論ともに所要時間が長く、モデルサイズが大きい傾向にある。ただしSqueezeNetは学習・推論ともに比較的短時間で処理を終えることができ、モデルサイズも小さかった。MobileNetはモデルサイズの割に学習とGPUを使った推論が遅く、実装上の問題でGPUを有効に使えていないように見える。  
ImageNet向けのモデルを元にフィルタ数を減らした場合、精度を落とさずに速度とモデルサイズを改善できることがあった。小さなデータセットに対しては、縮小版のモデルも選択肢として有効である。

Raspberry Pi向けのモデルは少ない計算コストで良い精度を示した。データセットや環境に特化したモデルを作ることは、計算コストと精度を両立する上で有効な手段と言える。
