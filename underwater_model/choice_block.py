from underwater_model.op import *
from underwater_model.trans_block_normal import TransformerBlock
from underwater_model.trans_block_dual import TransformerBlock_dual
from underwater_model.trans_block_eca import TransformerBlock_eca
from underwater_model.trans_block_sa import TransformerBlock_sa
from underwater_model.trans_block_sge import TransformerBlock_sge
class block_choice(nn.Module):
    def __init__(self, n_channels=128):
        super(block_choice, self).__init__()
        self.resblock_skipconnect = Identity(False)
        self.resblock_dil4Conv = DilConv(n_channels, n_channels, 3, 1, 4, 4, affine=True, upsample=False)
        self.resblock_conv = Conv(n_channels, n_channels, 3, 1, 1, affine=True, upsample=False)

        self.resblock_da = DoubleAttentionLayer(n_channels, n_channels, n_channels)

        self.resblock_se = SELayer(n_channels)

        self.resblock_trans_normal = nn.Sequential(*[TransformerBlock(dim=int(n_channels), num_heads=2, ffn_expansion_factor=2.66,
                         bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.resblock_trans_dual = nn.Sequential(
            *[TransformerBlock_dual(dim=int(n_channels), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.resblock_trans_eca = nn.Sequential(
            *[TransformerBlock_eca(dim=int(n_channels), num_heads=2, ffn_expansion_factor=2.66,
                                    bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.resblock_trans_sa = nn.Sequential(
            *[TransformerBlock_sa(dim=int(n_channels), num_heads=2, ffn_expansion_factor=2.66,
                                    bias=False, LayerNorm_type='WithBias') for i in range(1)])
        self.resblock_trans_sge = nn.Sequential(
            *[TransformerBlock_sge(dim=int(n_channels), num_heads=2, ffn_expansion_factor=2.66,
                                    bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, select):
        if select == 0:
            out = self.resblock_skipconnect(x)
        elif select == 1:
            out = self.resblock_dil4Conv(x)
        elif select == 2:
            out = self.resblock_conv(x)
        elif select == 3:
            out = self.resblock_se(x)
        elif select == 4:
            out = self.resblock_trans_normal(x)
        elif select == 5:
            out = self.resblock_trans_dual(x)
        elif select == 6:
            out = self.resblock_trans_eca(x)
        elif select == 7:
            out = self.resblock_trans_sa(x)
        elif select == 8:
            out = self.resblock_trans_sge(x)
        return out + x