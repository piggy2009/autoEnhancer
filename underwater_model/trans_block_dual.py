import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

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

##########################################################################
class TransformerBlock_dual(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_dual, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_dual(num_heads, int(dim / num_heads), int(dim / num_heads), int(dim / num_heads))
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

if __name__ == '__main__':
    input = torch.zeros([2, 48, 128, 128])
    # model = Restormer()
    # output = model(input)
    model2 = nn.Sequential(*[
        TransformerBlock_dual(dim=int(48), num_heads=2, ffn_expansion_factor=2.66,
                         bias=False, LayerNorm_type='WithBias') for i in range(1)])
    # model3 = Attention_sa(1, 16, 48)
    output2 = model2(input)
    print(output2.shape)