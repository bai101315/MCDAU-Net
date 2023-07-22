import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4  #1e-4  #1e-5


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()  # shape = [32, 64, 2000, 80]

        y = self.avg_pool(X_input)  # shape = [32, 64, 1, 1]
        y = y.view(b, c)  # shape = [32,64]

        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)  # shape = [32, 64] * [64, 4] = [32, 4]

        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y)  # shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)  # shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input * y.expand_as(X_input)



class M_Encoder(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False,
                 BatchNorm=False, num_groups=32,
                 res=False):
        super(M_Encoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode1 = ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)
        self.encode2 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)
        self.encode3 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)

        self.pooling = pooling

        contact_channels = output_channels * 3
        self.se = SE_Block(contact_channels)

        self.conv2 = M_Conv(contact_channels, output_channels, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

    def forward(self, x):

        out1 = self.encode1(x)
        out2 = self.encode2(out1)
        out3 = self.encode3(out2)

        out = torch.cat([out1, out2], dim=1)
        conv = torch.cat([out, out3], dim=1)


        _, ch, _, _ = conv.size()
        conv = self.se(conv)

        conv = self.conv2(conv)

        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            nn.ReLU(inplace=True)(pool)

            return conv, pool
        else:
            return conv, conv


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class M_Decoder(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2


        self.conv1 = ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups)
        self.conv2 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups)
        self.conv3 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups)


        contact_channels = 3 * output_channels + input_channels
        self.change = M_Conv(contact_channels, output_channels, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.se = SE_Block(contact_channels)

        ag_channels = int(input_channels / 2)
        self.ag = Attention_block(ag_channels, ag_channels, ag_channels)

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')

        x_big = self.ag(x_big, out)
        out = torch.cat([x_big,out], dim=1)

        # out = self.decode(out)

        out1 = self.conv1(out)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        conv1 = torch.cat([out1, out2], dim=1)
        conv2 = torch.cat([out, out3], dim=1)

        out = torch.cat([conv1,conv2], dim=1)

        _,ch,_,_ = out.size()
        conv = self.se(out)

        out = self.change(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, input_channels, output_channels, dilation, kernel_size=3, deconv=False, dowansample=None,
                 bn=False, BatchNorm=False, num_groups=32):
        super(Bottleneck, self).__init__()

        # padding0 = (dilation[0] * kernel_size - 1) // 2
        # padding1 = (dilation[1] * kernel_size - 1) // 2
        # padding2 = (dilation[2] * kernel_size - 1) // 2

        padding0 = dilation[0]
        padding1 = dilation[1]
        padding2 = dilation[2]

        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding0,
                         dilation=dilation[0],
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, is_relu=True, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding1,
                         dilation=dilation[1],
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, is_relu=True, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding2,
                         dilation=dilation[2],
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, is_relu=False, num_groups=num_groups),
        )
        self.conv = M_Conv(input_channels, output_channels, kernel_size=3,)
        self.downsample = dowansample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.encode(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        identity = self.conv(identity)

        out += identity

        out = self.relu(out)

        return out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class CCAF2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CCAF2, self).__init__()
        self.softamx = nn.Softmax2d()
        self.ch_conv = ConvBnRelu2d(in_channels * 2,out_channels,kernel_size=3,is_bn=True,is_relu=True,padding=(3-1)//2)

        self.conv1 = ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=2,padding=2,is_bn=True,is_relu=True)
        self.conv2 = ConvBnRelu2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4, is_bn=True,
                                  is_relu=True)

        self.se1 = SE_Block(32)
        self.se2 = SE_Block(64)
        self.Se = SE_Block((32+64) * 2)

    def forward(self,conv1,conv2):
        _, ch1, _, _ = conv1.size()
        _, ch2, _, _ = conv2.size()

        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)

        conv1 = self.se1(conv1)
        conv2 = self.se2(conv2)

        # gf1 = self.softamx(conv1)
        # gf2 = self.softamx(conv2)
        #
        # conv1 = conv1 * gf1
        # conv2 = conv2 * gf2

        out = torch.cat([conv1,conv2],dim=1)

        tmp1 = self.conv1(out)
        tmp2 = self.conv2(out)

        out = torch.cat([tmp1,tmp2],dim=1)

        _, ch, _, _ = out.size()
        out = self.Se(out)

        out = self.ch_conv(out)

        return out

class CCAF3(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(CCAF3, self).__init__()
        self.softamx = nn.Softmax2d()
        self.ch_conv=ConvBnRelu2d(in_channels * 3,out_channels,kernel_size=3,is_bn=True,is_relu=True,padding=(3-1)//2)

        self.se1 = SE_Block(32)
        self.se2 = SE_Block(64)
        self.se3 = SE_Block(128)
        self.Se = SE_Block((32+64+128) * 3)

        self.conv1 = ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=2,padding=2,is_bn=True,is_relu=True)
        self.conv2 = ConvBnRelu2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4, is_bn=True,
                                  is_relu=True)
        self.conv3 =ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=8,padding=8,is_bn=True,is_relu=True)


    def forward(self,conv1,conv2,conv3):
        _, ch1, _, _ = conv1.size()
        _, ch2, _, _ = conv2.size()
        _, ch3, _, _ = conv3.size()

        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)

        conv2 = F.max_pool2d(conv2, kernel_size=2, stride=2)



        conv1 = self.se1(conv1)
        conv2 = self.se2(conv2)
        conv3 = self.se3(conv3)

        # gf1 = self.softamx(conv1)
        # gf2 = self.softamx(conv2)
        # gf3 = self.softamx(conv3)
        # conv1 = conv1 * gf1
        # conv2 = conv2 * gf2
        # conv3 = conv3 * gf3

        out1 = torch.cat([conv1,conv2],dim=1)
        out = torch.cat([conv3,out1],dim=1)

        tmp1 = self.conv1(out)
        tmp2 = self.conv2(out)
        tmp3 = self.conv3(out)

        out = torch.cat([tmp1, tmp2], dim=1)
        out = torch.cat([out, tmp3], dim=1)

        _, ch, _, _ = out.size()
        out = self.Se(out)

        out = self.ch_conv(out)

        return out


class CCAF4(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(CCAF4, self).__init__()
        self.softamx = nn.Softmax2d()

        self.ch_conv = ConvBnRelu2d(in_channels * 4,out_channels,kernel_size=3,is_bn=True,is_relu=True,padding=(3-1)//2)

        self.se1 = SE_Block(32)
        self.se2 = SE_Block(64)
        self.se3 = SE_Block(128)
        self.se4 = SE_Block(128)
        self.Se = SE_Block((32+64+128+128)*4)

        self.conv1 = ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=1,padding=1,is_bn=True,is_relu=True)
        self.conv2 = ConvBnRelu2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2, is_bn=True,
                                  is_relu=True)
        self.conv3 = ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=4,padding=4,is_bn=True,is_relu=True)
        self.conv4 = ConvBnRelu2d(in_channels,in_channels,kernel_size=3,dilation=8,padding=8,is_bn=True,is_relu=True)

    def forward(self,conv1,conv2,conv3,conv4):
        _, ch1, _, _ = conv1.size()
        _, ch2, _, _ = conv2.size()
        _, ch3, _, _ = conv3.size()
        _, ch4, _, _ = conv4.size()

        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv1 = F.max_pool2d(conv1, kernel_size=2, stride=2)

        conv2 = F.max_pool2d(conv2, kernel_size=2, stride=2)
        conv2 = F.max_pool2d(conv2, kernel_size=2, stride=2)

        conv3 = F.max_pool2d(conv3, kernel_size=2, stride=2)
        # conv3 = F.max_pool2d(conv3, kernel_size=2, stride=2)

        conv1 = self.se1(conv1)
        conv2 = self.se2(conv2)
        conv3 = self.se3(conv3)
        conv4 = self.se4(conv4)

        # gf1 = self.softamx(conv1)
        # gf2 = self.softamx(conv2)
        # gf3 = self.softamx(conv3)
        # gf4 = self.softamx(conv4)
        #
        # conv1 = conv1 * gf1
        # conv2 = conv2 * gf2
        # conv3 = conv3 * gf3
        # conv4 = conv4 * gf4

        out1 = torch.cat([conv1,conv2],dim=1)
        out2 = torch.cat([conv3,conv4],dim=1)

        out = torch.cat([out1, out2], dim=1)

        tmp1 = self.conv1(out)
        tmp2 = self.conv2(out)
        tmp3 = self.conv3(out)
        tmp4 = self.conv4(out)

        out1 = torch.cat([tmp1, tmp2], dim=1)
        out2 = torch.cat([tmp3, tmp4], dim=1)

        out = torch.cat([out1, out2], dim=1)

        _, ch, _, _ = out.size()
        out = self.Se(out)

        out = self.ch_conv(out)

        return out

class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32, pooling=True):
        super(StackEncoder, self).__init__()
        padding=(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling=pooling

    def forward(self,x):
        y = self.encode(x)
        y_small = y
        if self.pooling:
            y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small

class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackDecoder, self).__init__()
        padding=(dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y