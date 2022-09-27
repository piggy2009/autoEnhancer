import torch
import torch.nn as nn
from underwater_model.base_SPOS import Base
from underwater_model.search_net import Search
from underwater_model.choice_block import block_choice

class Water(nn.Module):
    def __init__(self, dim):
        super(Water, self).__init__()

        self.base = Base(dim)

        self.align1 = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))

        self.align2 = nn.Sequential(nn.Conv2d((dim * 2 ** 1) * 2, (dim * 2 ** 1), kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.align3 = nn.Sequential(nn.Conv2d((dim * 2 ** 2) * 2, (dim * 2 ** 2), kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.align4 = nn.Sequential(nn.Conv2d((dim * 2 ** 3) * 2, (dim * 2 ** 3), kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.search = Search(dim)

        self.refine = block_choice(dim * 2 ** 1)

        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, 3, kernel_size=1, stride=1))

        self.de_predict_rgb = nn.Sequential(nn.Conv2d(dim * 2 ** 3, 3, kernel_size=1, stride=1))

        self.de_predict_lab = nn.Sequential(nn.Conv2d(dim * 2 ** 3, 3, kernel_size=1, stride=1))

    def forward(self, rgb, lab, select):
        level1_rgb, level2_rgb, level3_rgb, level4_rgb, \
        level1_lab, level2_lab, level3_lab, level4_lab, x_rgb_in, x_lab_in = self.base(rgb, lab, select[:8])

        level1 = self.align1(torch.cat([level1_rgb, level1_lab], dim=1))

        level2 = self.align2(torch.cat([level2_rgb, level2_lab], dim=1))

        level3 = self.align3(torch.cat([level3_rgb, level3_lab], dim=1))

        level4 = self.align4(torch.cat([level4_rgb, level4_lab], dim=1))

        mid_feat = self.search(level4, level3, level2, level1, select[8:11])
        mid_feat = self.refine(mid_feat, select[11])


        mid_rgb = self.de_predict_rgb(level4)
        mid_lab = self.de_predict_rgb(level4)

        final = self.de_predict(mid_feat)


        return final, mid_rgb, mid_lab

if __name__ == '__main__':
    a = torch.zeros(2, 3, 256, 256)
    b = torch.zeros(2, 128, 1, 1)

    model = Water(80)
    r = model(a, a, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])


    print(r[0].shape)



