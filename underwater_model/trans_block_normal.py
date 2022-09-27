import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# ECA attention module
class Attention_eca(nn.Module):
    def __init__(self, num_heads, k_size, bias):
        super(Attention_eca, self).__init__()
        self.num_heads = num_heads

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            y = self.avg_pool(head)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            out = head * y.expand_as(head)
            outputs.append(out)
        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output


# SGE attention module
class Attention_sge(nn.Module):
    def __init__(self, num_heads, groups):
        super(Attention_sge, self).__init__()
        self.num_heads = num_heads

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            b, c, h, w = head.size()
            # print(x.shape)
            head = head.reshape(b * self.groups, -1, h, w)
            xn = head * self.avg_pool(head)
            xn = xn.sum(dim=1, keepdim=True)
            t = xn.reshape(b * self.groups, -1)
            t = t - t.mean(dim=1, keepdim=True) + 1e-5
            std = t.std(dim=1, keepdim=True) + 1e-5
            t = t / std
            t = t.reshape(b, self.groups, h, w)
            t = t * self.weight + self.bias
            t = t.reshape(b * self.groups, 1, h, w)
            head = head * self.sig(t)
            head = head.reshape(b, c, h, w)
            outputs.append(head)

        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output


class Attention_sa(nn.Module):
    def __init__(self, num_heads, groups, channel):
        super(Attention_sa, self).__init__()
        self.num_heads = num_heads

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
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            b, c, h, w = head.shape

            head = head.reshape(b * self.groups, -1, h, w)
            x_0, x_1 = head.chunk(2, dim=1)

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

            outputs.append(out)

        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output

class Attention_dual(nn.Module):
    def __init__(self, num_heads, in_channels: int, c_m: int, c_n: int, reconstruct = False):
        super(Attention_dual, self).__init__()
        self.num_heads = num_heads

        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            batch_size, c, h, w = head.size()
            assert c == self.in_channels, 'input channel not equal!'
            A = self.convA(head)  # (B, c_m, h, w) because kernel size is 1
            B = self.convB(head)  # (B, c_n, h, w)
            V = self.convV(head)  # (B, c_n, h, w)
            tmpA = A.view(batch_size, self.c_m, h * w)
            attention_maps = B.view(batch_size, self.c_n, h * w)
            attention_vectors = V.view(batch_size, self.c_n, h * w)
            attention_maps = F.softmax(attention_maps, dim=-1)  # softmax on the last dimension to create attention maps
            # step 1: feature gathering
            global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
            # step 2: feature distribution
            attention_vectors = F.softmax(attention_vectors, dim=1)  # (B, c_n, h * w) attention on c_n dimension
            tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
            tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
            if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)

            outputs.append(tmpZ)

        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
