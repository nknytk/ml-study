# coding: utf-8

import chainer
from chainer import links as L
from chainer import functions as F


class InceptionResNetV2(chainer.ChainList):
    def __init__(self, dim_out=1000, base_filter_num=32, ablocks=5, bblocks=10, cblocks=5, dropout=0.2, scaling=0.1):
        layers = [Stem(base_filter_num)]
        for i in range(ablocks):
            layers.append(InceptionResNetBlockA(base_filter_num, scaling))
        layers.append(ReductionBlockA(base_filter_num*12, base_filter_num*8, base_filter_num*8, base_filter_num*12, base_filter_num*12))
        for i in range(bblocks):
            layers.append(InceptionResNetBlockB(base_filter_num, scaling))
        layers.append(ReductionBlockB(base_filter_num))
        for i in range(cblocks):
            layers.append(InceptionResNetBlockC(base_filter_num, scaling))
        layers.append(L.Linear(base_filter_num * 64, dim_out))
        super().__init__(*layers)
        self.dr_rate=dropout


    def predict(self, x, as_proba=False):
        h = x
        for i in range(len(self) - 1):
            h = self[i](h)
        h = F.average_pooling_2d(h, ksize=8)

        r = self[-1](F.dropout(h, self.dr_rate))
        if as_proba:
            r = F.softmax(r).data
        return r

    def __call__(self, x, t):
        pred = self.predict(x)
        loss = F.softmax_cross_entropy(pred, t)
        acc = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss


class ConvAct(chainer.link.Chain):
    def __init__(self, *args, **kwargs):
        oc = kwargs['out_channels'] if 'out_channels' in kwargs else args[1]
        super().__init__(
            conv=L.Convolution2D(*args, **kwargs),
            bn=L.BatchNormalization(oc)
        )

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))


class Stem(chainer.link.Chain):
    def __init__(self, base_filter_num=32):
        super().__init__(
            # 299x299 => (299 + 0 * 2 - 3) / 2 + 1 => 149x149
            conv1=ConvAct(in_channels=3, out_channels=base_filter_num, ksize=3, stride=2, pad=0),
            # 149x149 => (149 + 0 * 2 - 3) / 1 + 1 => 147x147
            conv2=ConvAct(in_channels=base_filter_num, out_channels=base_filter_num, ksize=3, stride=1, pad=0),
            # 147x147 => (147 + 1 * 2 - 3) / 1 + 1 => 147x147
            conv3=ConvAct(in_channels=base_filter_num, out_channels=base_filter_num*2, ksize=3, stride=1, pad=1),
            # 147x147 => (147 + 0 * 2 - 3) / 2 + 1 => 73x73
            conv4=ConvAct(in_channels=base_filter_num*2, out_channels=base_filter_num*3, ksize=3, stride=2, pad=0),
            # 73x73 => (73 + 0 * 2 - 1) / 1 + 1 => 73x73
            conv_a1=ConvAct(in_channels=base_filter_num*5, out_channels=base_filter_num*2, ksize=1, stride=1, pad=0),
            # 73x73 => (73 + 0 * 2 - 3) / 1 + 1 => 71x71
            conv_a2=ConvAct(in_channels=base_filter_num*2, out_channels=base_filter_num*3, ksize=3, stride=1, pad=0),
            # 73x73 => (73 + 0 * 2 - 1) / 1 + 1 => 73x73
            conv_b1=ConvAct(in_channels=base_filter_num*5, out_channels=base_filter_num*2, ksize=1, stride=1, pad=0),
            # 73x73 => (73 + 3 * 2 - 7) / 1 + 1 x (73 + 0 * 2 - 1) / 1 + 1 => 73x73
            conv_b2=ConvAct(in_channels=base_filter_num*2, out_channels=base_filter_num*2, ksize=(7, 1), stride=1, pad=(3, 0)),
            # 73x73 => (73 + 0 * 2 - 1) / 1 + 1 x (73 + 3 * 2 - 7) / 1 + 1 => 73x73
            conv_b3=ConvAct(in_channels=base_filter_num*2, out_channels=base_filter_num*2, ksize=(1, 7), stride=1, pad=(0, 3)),
            # 73x73 => (73 + 0 * 2 - 3) / 1 + 1 => 71x71
            conv_b4=ConvAct(in_channels=base_filter_num*2, out_channels=base_filter_num*3, ksize=3, stride=1, pad=0),
            # 71x71 => (71 + 0 * 2 - 3) / 2 + 1 => 35x35
            conv5=ConvAct(in_channels=base_filter_num*6, out_channels=base_filter_num*6, ksize=3, stride=2, pad=0)
        )

    def __call__(self, x):
        h =  self.conv1(x)
        h =  self.conv2(h)
        h =  self.conv3(h)

        h1 =  self.conv4(h)
        h2 = F.max_pooling_2d(h, ksize=(3, 3), stride=2, pad=0)
        h = F.concat((h1, h2), axis=1)

        ha =  self.conv_a1(h)
        ha =  self.conv_a2(ha)
        hb =  self.conv_b1(h)
        hb =  self.conv_b2(hb)
        hb =  self.conv_b3(hb)
        hb =  self.conv_b4(hb)
        h = F.concat((ha, hb), axis=1)

        h1 =  self.conv5(h)
        h2 = F.max_pooling_2d(h, ksize=3, stride=2, pad=0)
        h = F.concat((h1, h2), axis=1)
        # output dimensions: size=(35, 35), channels=base_filter_num*12
        return h


class InceptionResNetBlockA(chainer.link.Chain):
    def __init__(self, base_filter_num, scaling):
        super().__init__(
            conv_a=ConvAct(base_filter_num*12, base_filter_num, ksize=1, stride=1, pad=0),
            conv_b1=ConvAct(base_filter_num*12, base_filter_num, ksize=1, stride=1, pad=0),
            conv_b2=ConvAct(base_filter_num, base_filter_num, ksize=3, stride=1, pad=1),
            conv_c1=ConvAct(base_filter_num*12, base_filter_num, ksize=1, stride=1, pad=0),
            conv_c2=ConvAct(base_filter_num, int(base_filter_num * 1.5), ksize=3, stride=1, pad=1),
            conv_c3=ConvAct(int(base_filter_num * 1.5), base_filter_num * 2, ksize=3, stride=1, pad=1),
            conv_m=L.Convolution2D(base_filter_num*4, base_filter_num*12, ksize=1, stride=1, pad=0)
        )
        self.scaling = scaling

    def __call__(self, x):
        h_org = F.relu(x)

        ha = self.conv_a(h_org)
        hb = self.conv_b1(h_org)
        hb = self.conv_b2(hb)
        hc = self.conv_c1(h_org)
        hc = self.conv_c2(hc)
        hc = self.conv_c3(hc)

        hm = F.concat((ha, hb, hc), axis=1)
        hm = self.conv_m(hm)
        return h_org + hm * self.scaling


class ReductionBlockA(chainer.link.Chain):
    def __init__(self, in_channels, k, l, m, n):
        super().__init__(
            # 35x35 => (35 + 0 * 2 - 3) / 1 + 1 => 17x17
            conv_n=ConvAct(in_channels=in_channels, out_channels=n, ksize=3, stride=2, pad=0),
            # 35x35 => (35 + 0 * 2 - 1) / 1 + 1 => 35x35
            conv_k=ConvAct(in_channels=in_channels, out_channels=k, ksize=1, stride=1, pad=0),
            # 35x35 => (35 + 1 * 2 - 3) / 1 + 1 => 35x35
            conv_l=ConvAct(in_channels=k, out_channels=l, ksize=3, stride=1, pad=1),
            # 35x35 => (35 + 0 * 2 - 3) / 1 + 1 => 17x17
            conv_m=ConvAct(in_channels=l, out_channels=m, ksize=3, stride=2, pad=0),
        )

    def __call__(self, x):
        h1 = F.max_pooling_2d(x, ksize=(3, 3), stride=2)
        h2 =  self.conv_n(x)
        h3 =  self.conv_k(x)
        h3 =  self.conv_l(h3)
        h3 =  self.conv_m(h3)
        h = F.concat((h1, h2, h3), axis=1)
        return h


class InceptionResNetBlockB(chainer.link.Chain):
    def __init__(self, base_filter_num, scaling):
        super().__init__(
            conv_a=ConvAct(base_filter_num*36, base_filter_num*6, ksize=1, stride=1, pad=0),
            conv_b1=ConvAct(base_filter_num*36, base_filter_num*4, ksize=1, stride=1, pad=0),
            conv_b2=ConvAct(base_filter_num*4, base_filter_num*5, ksize=(1, 7), stride=1, pad=(0, 3)),
            conv_b3=ConvAct(base_filter_num*5, base_filter_num*6, ksize=(7, 1), stride=1, pad=(3, 0)),
            conv_m=L.Convolution2D(base_filter_num*12, base_filter_num*36, ksize=1, stride=1, pad=0)
        )
        self.scaling = scaling

    def __call__(self, x):
        h_org = F.relu(x)

        ha = self.conv_a(h_org)
        hb = self.conv_b1(h_org)
        hb = self.conv_b2(hb)
        hb = self.conv_b3(hb)

        hm = F.concat((ha, hb), axis=1)
        hm = self.conv_m(hm)
        return h_org + hm * self.scaling


class ReductionBlockB(chainer.link.Chain):
     def __init__(self, base_filter_num):
          super().__init__(
              conv_a1=ConvAct(base_filter_num*36, base_filter_num*8, ksize=1, stride=1, pad=0),
              conv_a2=ConvAct(base_filter_num*8, base_filter_num*12, ksize=3, stride=2, pad=0),
              conv_b1=ConvAct(base_filter_num*36, base_filter_num*8, ksize=1, stride=1, pad=0),
              conv_b2=ConvAct(base_filter_num*8, base_filter_num*8, ksize=3, stride=2, pad=0),
              conv_c1=ConvAct(base_filter_num*36, base_filter_num*8, ksize=1, stride=1, pad=0),
              conv_c2=ConvAct(base_filter_num*8, base_filter_num*8, ksize=3, stride=1, pad=1),
              conv_c3=ConvAct(base_filter_num*8, base_filter_num*8, ksize=3, stride=2, pad=0),
          )

     def __call__(self, x):
         hn = F.max_pooling_2d(x, ksize=3, stride=2, pad=0)
         ha = self.conv_a1(x)
         ha = self.conv_a2(ha)
         hb = self.conv_b1(x)
         hb = self.conv_b2(hb)
         hc = self.conv_c1(x)
         hc = self.conv_c2(hc)
         hc = self.conv_c3(hc)
         h = F.concat((hn, ha, hb, hc), axis=1)
         return h


class InceptionResNetBlockC(chainer.link.Chain):
    def __init__(self, base_filter_num, scaling):
        super().__init__(
            conv_a=ConvAct(base_filter_num*64, base_filter_num*6, ksize=1, stride=1, pad=0),
            conv_b1=ConvAct(base_filter_num*64, base_filter_num*6, ksize=1, stride=1, pad=0),
            conv_b2=ConvAct(base_filter_num*6, base_filter_num*7, ksize=(1, 3), stride=1, pad=(0, 1)),
            conv_b3=ConvAct(base_filter_num*7, base_filter_num*8, ksize=(3, 1), stride=1, pad=(1, 0)),
            conv_m=L.Convolution2D(base_filter_num*14, base_filter_num*64, ksize=1, stride=1, pad=0)
        )
        self.scaling = scaling

    def __call__(self, x):
        h_org = F.relu(x)

        ha = self.conv_a(h_org)
        hb = self.conv_b1(h_org)
        hb = self.conv_b2(hb)
        hb = self.conv_b3(hb)

        hm = F.concat((ha, hb), axis=1)
        hm = self.conv_m(hm)
        return h_org + hm * self.scaling
