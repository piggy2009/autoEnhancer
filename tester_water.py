import torch
from torchvision import transforms
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import calculate_psnr, calculate_ssim
import tqdm
import os
from PIL import Image, ImageCms
from torch.autograd import Variable
import numpy as np
from infer_water import read_testset
import cv2
import torch.nn.functional as F


# assert torch.cuda.is_available()

train_loader = None
device_id = 0

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args):
    # global train_loader

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()

    max_train_iters = args['max_train_iters']

    # print('clear bn statics....')
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.running_mean = torch.zeros_like(m.running_mean)
    #         m.running_var = torch.ones_like(m.running_var)

    # print('train bn with training set (BN sanitize) ....')
    # model.cuda(device_id).train()
    # dataloader_iterator = iter(train_loader)
    # for step in tqdm.tqdm(range(max_train_iters)):
    #     data = next(dataloader_iterator)
    #     inputs, flows, labels = data
    #     inputs = Variable(inputs).cuda(device_id)
    #     flows = Variable(flows).cuda(device_id)
    #     labels = Variable(labels).cuda(device_id)
    #     out1u, out2u, out2r, out3r, out4r, out5r = model(inputs, architecture=cand)
    #     # print('training:', step)
    #     del data, out1u, out2u, out2r, out3r, out4r, out5r

    print('starting test....')
    model.cuda(device_id).eval()
    image_names = read_testset(args['dataset'], args['image_path'])
    psnr_record = AvgMeter()
    ssim_record = AvgMeter()

    factor = 8
    for name in image_names:

        # img_list = [i_id.strip() for i_id in open(imgs_path)]
        img = Image.open(os.path.join(args['image_path'], name + '.png')).convert('RGB')

        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = np.array(img)
        # img = cv2.resize(img, (256, 256))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
        lab_var = Variable(img_transform(lab).unsqueeze(0), volatile=True).cuda()


        h, w = img_var.shape[2], img_var.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        img_var = F.pad(img_var, (0, padw, 0, padh), 'reflect')
        lab_var = F.pad(lab_var, (0, padw, 0, padh), 'reflect')
        #
        # # temp = (1, 1, 0)
        prediction, _, _ = model(img_var, lab_var, cand)
        prediction = prediction[:, :, :h, :w]


        prediction = torch.clamp(prediction, 0, 1)
        prediction = prediction.permute(0, 2, 3, 1).cpu().detach().numpy()
        prediction = np.squeeze(prediction)

        gt = Image.open(os.path.join(args['gt_path'], name + '.png')).convert('RGB')
        gt = np.asarray(gt)
        # gt = cv2.resize(gt, (256, 256))
        # print(gt.shape, '-----', prediction.shape)
        psnr = calculate_psnr(prediction * 255.0, gt)
        ssim = calculate_ssim(prediction * 255.0, gt)


        psnr_record.update(psnr)
        ssim_record.update(ssim)


    print('psnr: {:.5f} ssim: {:.5f}'.format(psnr_record.avg, ssim_record.avg))

    return psnr_record.avg, ssim_record.avg

