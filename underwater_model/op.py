import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none': lambda in_C, out_C, stride, upsample, affine: Zero(stride, upsample=upsample),
    'skip_connect': lambda in_C, out_C, stride, upsample, affine: Identity(upsample=upsample),
    'sep_conv_3x3': lambda in_C, out_C, stride, upsample, affine: SepConv(in_C, out_C, 3, stride, 1, affine=affine,
                                                               upsample=upsample),
    'sep_conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: SepConvDouble(in_C, out_C, 3, stride, 1,
                                                                                    affine=affine, upsample=upsample),
    'dil_conv_3x3': lambda in_C, out_C, stride, upsample, affine: DilConv(in_C, out_C, 3, stride, 2, 2, affine=affine,
                                                                          upsample=upsample),
    'dil_conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: DilConvDouble(in_C, out_C, 3, stride, 2, 2,
                                                                                    affine=affine, upsample=upsample),
    'dil_conv_3x3_dil4': lambda in_C, out_C, stride, upsample, affine: DilConv(in_C, out_C, 3, stride, 4, 4,
                                                                               affine=affine, upsample=upsample),

    'conv_3x3': lambda in_C, out_C, stride, upsample, affine: Conv(in_C, out_C, 3, stride, 1, affine=affine,
                                                                   upsample=upsample),
    'conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: ConvDouble(in_C, out_C, 3, stride, 1, affine=affine,
                                                                             upsample=upsample),

    'SpatialAttention': lambda in_C, out_C, stride, upsample, affine: SpatialAttention(in_C, 7),
    'ChannelAttention': lambda in_C, out_C, stride, upsample, affine: ChannelAttention(in_C, 16),

}
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct = False):
        """
        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)
        Returns
        -------
        """
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ

class sge_layer(nn.Module):
    def __init__(self, groups=64):
        super(sge_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        x = x.contiguous().view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.contiguous().view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True) + 1e-5
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.contiguous().view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.contiguous().view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.contiguous().view(b, c, h, w)
        return x


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


def conv3x3(in_planes, out_planes, stride):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // ratio)
        self.linear_2 = nn.Linear(self.in_channels // ratio, self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()
        out  = torch.mul(input_, feats)
        out = F.relu(self.bn1(self.conv1(out)), inplace=True)

        return out


class SpatialAttention(nn.Module):
    def __init__(self,in_C, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.in_channels = in_C
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv11 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn11 = nn.BatchNorm2d(self.in_channels)
    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out  = input * x

        out = F.relu(self.bn11(self.conv11(out)), inplace=True)

        return out


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(Conv, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class ConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(ConvDouble, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, upsample, affine=True):
        super(DilConv, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class DilConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, upsample, affine=True):
        super(DilConvDouble, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(SepConv, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0

        self.up = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        out = self.conv_1(x)
        out = self.bn(out)
        return out


class SepConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(SepConvDouble, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)



class Identity(nn.Module):

    def __init__(self, upsample):
        super(Identity, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        if self.upsample == True:
            x = self.up(x)
        return torch.relu(x)


class Zero(nn.Module):

    def __init__(self, stride, upsample):
        super(Zero, self).__init__()
        self.stride = stride
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        if self.upsample == True:
            x = self.up(x)
        else:
            x = x.mul(0.)
        return x
        # return x[:,:,::self.stride,::self.stride].mul(0.)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.conv_align = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        #                                 nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        temp = x * y.expand_as(x) + x
        return temp

if __name__ == '__main__':
    model = sa_layer(32, 4)
    input = torch.zeros(2, 32, 32, 32)
    output = model(input)
    print(output.shape)