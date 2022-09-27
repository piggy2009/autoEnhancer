import numpy as np
import os
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
from math import exp
# import pydensecrf.densecrf as dcrf
import torchvision
import torch.nn.functional as F
torch_ver = torch.__version__[:3]
from utils.ptcolor import *


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def quantAB(bins, vmax,vmin):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat=torch.cartesian_prod(a,a)
    return mat.view(1,bins**2,2,1,1)


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class Lab_Loss(nn.Module):
    def __init__(self, alpha=1,weight=1,levels=7,vmin=-80,vmax=80):
        super(Lab_Loss, self).__init__()
        self.alpha=alpha
        self.weight=weight
        self.levels=levels
        self.vmin=vmin
        self.vmax=vmax

    def Hist_2_Dist_L(self,img, tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2)
        p=torch.functional.softmax(-alpha*img_dist,dim=1)
        return p

    def Hist_2_Dist_AB(self,img,tab,alpha):
        img_dist=((img.unsqueeze(1)-tab)**2).sum(2)
        p = torch.nn.functional.softmax(-alpha*img_dist, dim=1)
        return p

    def loss_ab(self,img,gt,alpha,tab,levels):
        p= self.Hist_2_Dist_AB(img, tab,alpha).cuda()
        q= self.Hist_2_Dist_AB(gt,tab,alpha).cuda()
        p = torch.clamp(p, 0.001, 0.999)
        loss = -(q*torch.log(p)).sum([1,2,3]).mean()
        return loss

    def forward(self,img,gt):
	    tab=quantAB(self.levels,self.vmin,self.vmax).cuda()
	    lab_img=torch.clamp(rgb2lab(img),self.vmin,self.vmax)
	    lab_gt=torch.clamp(rgb2lab(gt),self.vmin,self.vmax)

	    loss_l=torch.abs(lab_img[:,0,:,:]-lab_gt[:,0,:,:]).mean()
	    loss_AB=self.loss_ab(lab_img[:,1:,:,:],lab_gt[:,1:,:,:],self.alpha,tab,self.levels)
	    loss=loss_l+self.weight*loss_AB
	    #return (loss,loss_l,loss_AB)
	    return loss

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.dsn_weight = dsn_weight

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        # print(preds[0].size())
        # print(target.size())

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=_pretrained_).features
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        pred = (pred - self.mean) / self.std
        true = (true - self.mean) / self.std
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)

class CriterionKL(nn.Module):
    def __init__(self):
        super(CriterionKL, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, target):
        assert preds.size() == target.size()

        n, c, w, h = preds.size()
        softmax_preds = F.softmax(target.permute(0, 2, 3, 1).contiguous().view(-1, c), dim=1)
        loss = (torch.sum(-softmax_preds * self.log_softmax(preds.permute(0, 2, 3, 1).contiguous().view(-1, c)))) / w / h

        return loss

class CriterionKL2(nn.Module):
    def __init__(self):
        super(CriterionKL2, self).__init__()

    def forward(self, preds, target):
        assert preds.size() == target.size()

        b, c, w, h = preds.size()
        preds = F.softmax(preds.view(b, -1), dim=1)
        target = F.softmax(target.view(b, -1), dim=1)
        loss = (preds * (preds / target).log()).sum() / b

        return loss

class CriterionStructure(nn.Module):
    def __init__(self):
        super(CriterionStructure, self).__init__()
        self.gamma = 2
        self.ssim = SSIM(window_size=11,size_average=True)

    def forward(self, pred, target):
        assert pred.size() == target.size()

        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        ##### focal loss #####
        p_t = torch.exp(-wbce)
        f_loss = (1 - p_t) ** self.gamma * wbce

        ##### ssim loss #####
        # ssim = 1 - self.ssim(pred, target)

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (1 * wbce + 0.8 * wiou + 0.05 * f_loss).mean()

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class CriterionKL3(nn.Module):
    def __init__(self):
        super(CriterionKL3, self).__init__()

    def KLD(self, input, target):
        input = input / torch.sum(input)
        target = target / torch.sum(target)
        eps = sys.float_info.epsilon
        return torch.sum(target * torch.log(eps + torch.div(target, (input + eps))))

    def forward(self, input, target):
        assert input.size() == target.size()

        return _pointwise_loss(lambda a, b:self.KLD(a,b), input, target)

class CriterionPairWise(nn.Module):
    def __init__(self, scale):
        super(CriterionPairWise, self).__init__()
        self.scale = scale

    def L2(self, inputs):
        return (((inputs ** 2).sum(dim=1)) ** 0.5).reshape(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]) + 1e-8

    def similarity(self, inputs):
        inputs = inputs.float()
        tmp = self.L2(inputs).detach()
        inputs = inputs / tmp
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        return torch.einsum('icm, icn->imn', [inputs, inputs])

    def sim_dis_compute(self, preds, targets):
        sim_err = ((self.similarity(targets) - self.similarity(preds)) ** 2) / ((targets.size(-1) * targets.size(-2)) ** 2) / targets.size(0)
        sim_dis = sim_err.sum()
        return sim_dis

    def forward(self, preds, targets):
        total_w, total_h = preds.shape[2], preds.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        max_pooling = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        loss = self.sim_dis_compute(max_pooling(preds), max_pooling(targets))
        return loss

class CriterionDice(nn.Module):
    def __init__(self):
        super(CriterionDice, self).__init__()

    def forward(self, pred, target):
        n = target.size(0)
        smooth = 1
        pred = F.sigmoid(pred)
        pred_flat = pred.view(n, -1)
        target_flat = target.view(n, -1)

        intersection = pred_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / n

        return loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

import torch
from torch import nn


class PixelInfoNCELoss(nn.Module):
    def __init__(
        self, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ) -> None:
        super().__init__()
        self.sclloss = SCLLoss(temperature, contrast_mode, base_temperature)

    def forward(self, features, labels):
        n, c, h, w = features.shape
        features = features.view(n, c, -1).transpose(2, 1).reshape(n, -1, 1, c)
        labels = labels.reshape(labels.shape[0], -1)
        labels[labels > 0] = 1
        labels = labels.reshape(labels.shape[0], -1)
        features = F.normalize(features, dim=-1)

        return self.sclloss(features, labels)


class SCLLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SCLLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_samples, n_views, ...].
            labels: ground truth of shape [bsz, n_samples].
            mask: contrastive mask of shape [bsz, n_samples, n_samples],
            mask_{n, i,j}=1 if sample j has the same class as sample i.
            Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) != 4:
            raise ValueError(
                "`features` needs to be [bsz, n_samples, n_views, ...].,"
                "features of 4 dimensions are required"
            )

        batch_size = features.shape[0]
        n_samples = features.shape[1]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(n_samples, dtype=torch.float32).to(device)
            mask = mask.repeat(batch_size, 1, 1)
        elif labels is not None:
            labels = labels.contiguous().view(batch_size, -1, 1)
            if labels.shape[1] != n_samples:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.transpose(1, 2)).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[2]
        # contrast_feature = torch.cat(torch.unbind(features, dim=2), dim=1)
        contrast_feature = features.reshape(batch_size, -1, features.shape[-1])
        if self.contrast_mode == "one":
            anchor_feature = features[:, :, :1]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        # print(anchor_feature)
        anchor_dot_contrast = torch.div(
            torch.bmm(anchor_feature, contrast_feature.transpose(1, 2)),
            self.temperature,
        )
        # print(f"{anchor_dot_contrast=}")
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        # mask = mask.repeat(1, anchor_count, contrast_count)
        mask = mask.repeat_interleave(anchor_count, dim=1)
        mask = mask.repeat_interleave(contrast_count, dim=2)

        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     2,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0,
        # )
        logits_mask = torch.ones_like(mask) - torch.eye(
            mask.shape[1], device=device
        ).repeat(batch_size, 1, 1)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(2, keepdim=True))
        # print(f"{exp_logits=}")
        # print(f"{log_prob=}")
        # compute mean of log_head=4-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(2) / mask.sum(2)
        # mean_log_prob_pos = ((mask * log_prob).sum(2) / mask.sum(2))[:, -1:]
        # print(f"{mean_log_prob_pos=}")
        # n_samples = 1

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size, n_samples * anchor_count).mean()
        # print(loss.item())
        return loss


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
# def crf_refine(img, annos):
#     def _sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     assert img.dtype == np.uint8
#     assert annos.dtype == np.uint8
#     assert img.shape[:2] == annos.shape
#
#     # img and annos should be np array with data type uint8
#
#     EPSILON = 1e-8
#
#     M = 2  # salient or not
#     tau = 1.05
#     # Setup the CRF model
#     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
#
#     anno_norm = annos / 255.
#
#     n_energy = -np.log_head=4((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
#     p_energy = -np.log_head=4(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))
#
#     U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
#     U[0, :] = n_energy.flatten()
#     U[1, :] = p_energy.flatten()
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
#
#     # Do the inference
#     infer = np.array(d.inference(1)).astype('float32')
#     res = infer[1, :]
#
#     res = res * 255
#     res = res.reshape(img.shape[:2])
#     return res.astype('uint8')

if __name__ == '__main__':
    # pixel_wise_loss = CriterionKL3()
    # pair_wise_loss = CriterionPairWise(scale=0.5)
    # ppa_wise_loss = CriterionStructure()
    # preds = torch.rand([2, 64, 24, 24])
    b = np.load('feat5.npy')
    preds = torch.from_numpy(b)
    # print(preds)
    # print(torch.sum(F.softmax(preds, dim=1)))
    # targets = torch.randn([2, 1, 24, 24])
    # targets = (targets > 0).type(torch.uint8)
    a = np.load('labels_contra.npy')
    targets = torch.from_numpy(a)
    from matplotlib import pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(b[0, 20])
    plt.subplot(1, 2, 2)
    plt.imshow(a[0, 0])
    plt.show()
    # loss = pixel_wise_loss(F.sigmoid(preds), F.sigmoid(preds))
    # loss = F.kl_div(preds, preds)
    # loss2 = pair_wise_loss(preds, targets)
    # print(ppa_wise_loss(preds, targets))
    net = PixelInfoNCELoss()
    loss = net(preds, targets)
    print(loss)
    # print(loss2)
