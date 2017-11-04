# 性能検証レポート

## 目的

今回実装したinception-V4とinception-resnet-V2の精度とコストを確かめる。気になるのは以下の3点  

* 実装が間違っていて精度が出ない、ということがないか
* その他の有名どころのモデルと比べて精度が高いのか
* ImageNetのような大規模なデータセットではなく、個人が扱えるような小規模なデータセットでも有効なものか

## 評価方法

### データセット

[17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)の画像を、各カテゴリ80枚中70枚を学習用、10枚を評価用に分けて使用する。  
オリジナル画像のサイズは揃っていないため、短辺に合わせて長辺の両端を切り落とし、CNNの入力用にリサイズする。

### 評価環境

| 項目 | 値 |
| ---- | --- |
| CPU  | Ryzen 1700 (8C16T) |
| GPU  | GeForce GTX 1070 |
| Memory | 16GB |
| Disk | M.2 SSD 256GB |
| OS | Ubuntu 16.04 |
| Cuda | 8.0 |
| Python | 3.5.2 |
| chainer | 3.0.0 |
| cupy | 2.0.0 |

### 評価軸

100epoch学習させ、精度と計算コストを確認する。

* 精度: 評価データに対する正解率の最高値(%)
* 学習所要時間: GPUを使用し、100epochの学習と評価を完了させるのにかかった時間(秒)
* 推論速度: CPUを1Core使用(`環境変数OMP_NUM_THREADS=1`を設定)し、1枚の画像を分類するのにかかる時間(秒)
* GPUメモリ: 専有したGPUのメモリ(MB)
* モデルサイズ: 学習済モデルをserializeして保存した時のnpzファイルサイズ(MB)

### モデル

| モデル | 入力画像サイズ | モデル説明 |
| --- | --- | --- |
| VGGNetBN | 224x224 | VGGNetに収束が早くなるようBatchNormalizationを追加　| 
| VGGNetBNQ | 224x224 | VGGNetBNのフィルタ数を1/4に減らした　| 
| GoogLeNetBN | 224x224 | [chainer公式のexpample](https://github.com/chainer/chainer/tree/master/examples/imagenet) の出力クラス数だけを17に変更 |
| ResNet50 | 224x224 | [chainer公式のexpample](https://github.com/chainer/chainer/tree/master/examples/imagenet) の出力クラス数だけを17に変更 |
| InceptionV4 | 299x299 | 今回実装したinception-v4 パラメータ数も論文のまま |
| InceptionV4_S | 299x299 | inception-v4を大幅に縮小 |
| InceptionResNetV2 | 299x299 | 今回実装したinception-resnet-v2 パラメータ数も論文のまま |
| InceptionResnetV2_S | 299x299 | inception-resnet-v2を大幅に縮小 |
| FaceClassifier100x100V | 100x100 | Raspberry Piでの使用を目指し、小規模データセット下での軽量さと精度の両立を図って実装した簡素なモデル。軽量性重視版 |
| FaceClassifier100x100V2 | 100x100 | Raspberry Piでの使用を目指し、小規模データセット下での軽量さと精度の両立を図って実装した簡素なモデル。精度重視版 |

## 結果

| モデル | 精度 | 学習所要時間  | 推論速度 | GPUメモリ | モデルサイズ | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| VGGNetBN | 12.9 | 2726 | 1.223 | | 3464 | 476 | 未収束 |
| VGGNetBNQ | 58.8 | 490 | 0.119 | 701 | 30 | 未収束 |
| GoogLeNetBN | 85.8 | 1844 | 0.248 | 1065 | 52 | |
| ResNet50 | 80.0 | 1268 | 0.369 | 1611 | 84 | |
| InceptionV4 | 79.4 | 3921 | 1.119 | 2911 | 147 | 未収束 |
| InceptionV4_S | 87.6 | 1458 | 0.067 | 523 | 1.9 | |
| InceptionResNetV2 | 89.4 | 3324 | 0.816 | 2445 | 105 | |
| InceptionResNetV2_S | 88.8 | 1179 | 0.073 | 603 | 2.7 | |
| FaceClassifier100x100V | 82.3 | 170 | 0.014 | 323 | 0.65 | |
| FaceClassifier100x100V2 | 87.0 | 201 | 0.027 | 353 | 4.2 | |

試したモデルの中ではinception-resnet-v2が最も高い精度を示した。実装は間違っていなさそうである。  
ImageNet向けのモデルは軒並み計算コストが高く、中でもinception-v4とinception-resnet-v2は論文通りのパラメータ数では計算コストが非常に高い。加えてinception-v4とVGGNetは収束が遅く、100epoch時点ではtrain lossが下がりきっていなかった。  
小規模データセットに対してImageNet向けのモデルは軒並みパラメータ過多であり、パラメータ数を減らすよう調整したほうが良い。ネットワーク構造が複雑なモデルはパラメータを減らした場合でも行列計算の回数が増える分計算に時間がかかるが、高い精度とモデルサイズの小ささを両立できる。

Raspberry Pi向けのモデルは少ない計算コストで良い精度を示した。データセットや環境に特化したモデルを作ることは、計算コストと精度を両立する上で有効な手段と言える。
