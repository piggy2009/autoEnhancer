import torch
from underwater_model.op import *
from underwater_model.choice_block import block_choice

class Search(nn.Module):
    def __init__(self, dim):
        super(Search, self).__init__()
        # round 1(1~3), level 1(1~3), op:1~12
        self.conv_up3 = nn.Sequential(nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)

        self.decoder_block3 = block_choice(n_channels=dim * 2 ** 2)
        self.decoder_block2 = block_choice(n_channels=dim * 2 ** 1)
        self.decoder_block1 = block_choice(n_channels=dim * 2 ** 1)

    def forward(self, level4, level3, level2, level1, select):
        # print(select)
        de_level3 = self.conv_up3(level4)
        de_level3 = torch.cat([de_level3, level3], 1)
        de_level3 = self.conv_cat3(de_level3)
        de_level3 = self.decoder_block3(de_level3, select[0])

        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, level2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, select[1])

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, level1], 1)
        de_level1 = self.decoder_block1(de_level1, select[2])

        return de_level1

if __name__ == '__main__':
    level1 = torch.zeros(2, 48, 128, 128)
    level2 = torch.zeros(2, 96, 64, 64)
    level3 = torch.zeros(2, 192, 32, 32)
    level4 = torch.zeros(2, 384, 16, 16)

    model = Search(dim=48)
    choice = [0, 0, 0]
    output = model(level4, level3, level2, level1, choice)
    print(output.size())