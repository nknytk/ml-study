機械学習の勉強に使ったコードを置くリポジトリ。  
主にPython3で実装される。

### cnn

#### inception_v4.py

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (arXiv:1602.07261)](https://arxiv.org/abs/1602.07261)で提案されたInception-v4のchainer実装。`InceptionV4`クラスを使用する。

学習データ量や使えるマシンリソースに応じてネットワークサイズを縮小できるよう、フィルタ数と層の数を初期化時にパラメータで指定できるように実装した。デフォルト設定で論文と同じ値になる。入力画像サイズは3x299x299で固定。

| パラメータ名 | 意味 | デフォルト値 |
| --- | --- | --- |
| dim_out | 出力値の次元数(クラス数) | 1000 |
| base_filter_num | convolutionのフィルタ数の多さ | 32 |
| ablocks | 論文Figure 9の`Inception-A`の繰り返し数 | 4 |
| bblocks | 論文Figure 9の`Inception-B`の繰り返し数 | 7 |
| cblocks | 論文Figure 9の`Inception-C`の繰り返し数 | 3 |
| dropout | 全結合層手前で行うdropoutの割合。keep=1-dropout | 0.2 |

#### inception_resnet_v2.py

[(arXiv:1602.07261)](https://arxiv.org/abs/1602.07261)で提案されたInception-ResNet-v2のchainer実装。`InceptionResNetV2`を使用する。 
batch normalizationを行う位置を論文から正確に理解できず、想像で実装した部分がある。また、Inception-ResNet-v2で使用されたReduction-Bの構造について論文に明記されていないが、channel数からInception-ResNet-v1("wider"ではない方)と同じであると推察して実装した。  
オリジナルの実装を再現できていない可能性が高い。

学習データ量や使えるマシンリソースに応じてネットワークサイズを縮小できるよう、フィルタ数と層の数を初期化時にパラメータで指定できるように実装した。デフォルト設定で論文と同じ値になる。入力画像サイズは3x299x299で固定。

| パラメータ名 | 意味 | デフォルト値 |
| --- | --- | --- |
| dim_out | 出力値の次元数(クラス数) | 1000 |
| base_filter_num | convolutionのフィルタ数の多さ | 32 |
| ablocks | 論文Figure 15の`Inception-resnet-A`の繰り返し数 | 5 |
| bblocks | 論文Figure 15の`Inception-resnet-B`の繰り返し数 | 10 |
| cblocks | 論文Figure 15の`Inception-resnet-C`の繰り返し数 | 5 |
| dropout | 全結合層手前で行うdropoutの割合。keep=1-dropout | 0.2 |
| scaling | Figure 20のscalingの定数 | 0.1 | 

層を浅くする時はscalingの値を1に近づけるよう調整したほうが良さそうな気がする

### report.md

上記CNNの性能検証レポート


## License

本リポジトリ内のコードとドキュメントについて、著作権と責任を完全に放棄します。
