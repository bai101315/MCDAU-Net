import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *


class UNet (nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(UNet, self).__init__()

        #1024
        # self.down2 = StackEncoder( 3,   64, kernel_size=3)   #256
        # self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        # self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        # self.down5 = StackEncoder(256,  512, kernel_size=3)   #32

        self.down2 = StackEncoder( 3,   32, kernel_size=3)   #256
        self.down3 = StackEncoder( 32,  64, kernel_size=3)   #128
        self.down4 = StackEncoder(64,  128, kernel_size=3, pooling=False)   #64

        self.center = nn.Sequential(
            ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )
        # 16
        # x_big_channels, x_channels, y_channels
        # self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 128,128, 64, kernel_size=3)  # 64
        self.up3 = StackDecoder( 64, 64,  32, kernel_size=3)  #128
        self.up2 = StackDecoder( 32, 32, 32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        # down5,out = self.down5(out)   #;print('down5',down5.size())
                                      #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)

        # out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        # out = torch.squeeze(out, dim=1)
        return out

class MCDAU_Net(nn.Module):
    def __init__(self, n_classes, bn=True, BatchNorm=False):
        super(MCDAU_Net, self).__init__()

        # the down convolution contain concat operation   两次卷积 +
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res= True)  # 512
        self.down2 = M_Encoder(32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res = True)  # 256
        self.down3 = M_Encoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, res= True)  # 128

        # the center  级联卷积 + ASPP
        self.bn = bn
        self.BatchNorm = BatchNorm

        # the center  级联卷积 + ASPP
        self.center1 = Bottleneck(128, 128, [1, 2, 1], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)
        self.center2 = Bottleneck(128, 128, [2, 4, 2], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)
        self.center3 = Bottleneck(128, 128, [4, 8, 4], kernel_size=3, bn=self.bn, BatchNorm=self.BatchNorm)

        self.aspp = ASPP(128, [4, 8, 4])

        # the up convolution contain concat operation
        self.up6 = M_Decoder(256, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.change1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.change2 = nn.Conv2d(in_channels=128 , out_channels=128, kernel_size=3, padding=1)
        self.change3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.change4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.ccaf2 = CCAF2(32+64,64)
        self.ccaf3 = CCAF3(32+64+128,128)
        self.ccaf4 = CCAF4(32+64+128+128,128)


        self.AG1= Attention_block(32, 32, 32*2)
        self.AG2 = Attention_block(64, 64,  64* 2)
        self.AG3 = Attention_block(32, 32, 32 * 2)

    def forward(self, x):
        _, _, img_shape, _ = x.size()

        conv1, out = self.down1(x) # 32,32

        conv2, out = self.down2(out) #64，64

        conv3, out = self.down3(out) #128， 128

        # out = self.center(out)
        out = self.center1(out)
        out = self.center2(out)
        out = self.center3(out)
        out = self.aspp(out)

        tmp1 = conv1
        tmp2 = self.ccaf2(conv1, conv2)
        tmp3 = self.ccaf3(conv1,conv2,conv3)

        out = self.ccaf4(conv1, conv2, conv3, out)

        up6 = self.up6(tmp3, out) # 128，128 ->64
        up7 = self.up7(tmp2, up6) # 64, 64 ->32
        up8 = self.up8(tmp1, up7) # 32,32 ->16

        # 上采样
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_8 = self.side_8(side_8)

        return side_8