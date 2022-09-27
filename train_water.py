import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn import functional as F
from matplotlib import pyplot as plt

import contextual_loss as cl

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path, saving_path
from water_dataset import WaterImageFolder, WaterImage2Folder, WaterImage3Folder, WaterImage4Folder
from underwater_model.model_SPOS import Water

from misc import AvgMeter, check_mkdir, VGGPerceptualLoss, Lab_Loss, GANLoss, VGG19_PercepLoss
from torch.backends import cudnn
import time
from utils.utils_mine import load_part_of_model
import random
import numpy as np

cudnn.benchmark = True

device_id = 0
device_id2 = 0

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(2021)
np.random.seed(2021)
# torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = saving_path
exp_name = 'WaterEnhance' + '_' + time_str

args = {
    'choice': 9,
    'layers': 12,
    'en_channels': [64, 128, 256],
    'dim': 48,
    'distillation': False,
    'L2': False,
    'KL': True,
    'iter_num': 200000,
    'iter_save': 4000,
    'iter_start_seq': 0,
    'train_batch_size': 3,
    'last_iter': 0,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.925,
    'snapshot': '',
    'pretrain': '',
    # 'imgs_file': '/home/user/ubuntu/data/LOLdataset/our485',
    'imgs_file': '/home/user/ubuntu/data/LSUI',
    'image_size': 320,
    'crop_size': [256, 320],
    # 'self_distill': 0.1,
    # 'teacher_distill': 0.6
}

imgs_file = os.path.join(datasets_root, args['imgs_file'])
# imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

joint_transform = joint_transforms.Compose([
    # joint_transforms.ImageResize(args['image_size']),
    joint_transforms.RandomCrop(args['crop_size'][0]),
    joint_transforms.RandomHorizontallyFlip(),
])

joint_transform2 = joint_transforms.Compose([
    # joint_transforms.ImageResize(args['image_size']),
    joint_transforms.RandomCrop(args['crop_size'][1]),
    joint_transforms.RandomHorizontallyFlip(),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
train_set = WaterImage2Folder(args['imgs_file'], joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

train_set2 = WaterImage2Folder(args['imgs_file'], joint_transform2, img_transform, target_transform)
train_loader2 = DataLoader(train_set2, batch_size=args['train_batch_size']-2, num_workers=4, shuffle=True)
# if train_set2 is not None:
#     train_loader2 = DataLoader(train_set2, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_perceptual = VGGPerceptualLoss().cuda()


log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main():

    net = Water(dim=args['dim']).cuda(device_id).train()
    net.apply(weights_init)

    remains = []
    for name, param in net.named_parameters():
        remains.append(param)
    # fix_parameters(net.named_parameters())
    # optimizer = optim.SGD([
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #      'lr': 2 * args['lr']},
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #      'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # ], momentum=args['momentum'])

    optimizer = optim.Adam([{'params': remains, 'lr': args['lr']}],
                         betas=(0.9, 0.999))
    # optimizer_d = optim.Adam([{'params': discriminator.parameters()}],
    #                        lr=args['lr'], betas=(0.9, 0.999))
    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 0.5 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
        optimizer.param_groups[2]['lr'] = args['lr']

    if len(args['pretrain']) > 0:
        print('pretrain model from ' + args['pretrain'])
        net = load_part_of_model(net, args['pretrain'], device_id=device_id)
        # fix_parameters(student.named_parameters())

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                  ) ** args['lr_decay']
            # optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                 ) ** args['lr_decay']
            # optimizer.param_groups[2]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']

            rgb, hsv, lab, target, lab_target = data
            train_single2(net, rgb, lab, target, lab_target, optimizer, curr_iter)
            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))


            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))

                return


def train_single2(net, rgb, lab, target, lab_target, optimizer, curr_iter):
    rgb = Variable(rgb).cuda(device_id)

    lab = Variable(lab).cuda(device_id)

    labels = Variable(target).cuda(device_id)
    labels_lab = Variable(lab_target).cuda(device_id)


    get_random_cand = lambda: tuple(np.random.randint(args['choice']) for i in range(args['layers']))

    optimizer.zero_grad()

    final, mid_rgb, mid_lab = net(rgb, lab, get_random_cand())

    loss0 = criterion(final, labels)
    loss1 = criterion_l1(final, labels)

    labels_rgb = F.interpolate(labels, size=mid_rgb.shape[2:], mode='bilinear')
    labels_lab = F.interpolate(labels_lab, size=mid_rgb.shape[2:], mode='bilinear')

    loss0_2 = criterion(mid_rgb, labels_rgb)
    loss1_2 = criterion_l1(mid_rgb, labels_rgb)


    loss0_lab = criterion(mid_lab, labels_lab)
    loss1_lab = criterion_l1(mid_lab, labels_lab)


    loss7 = criterion_perceptual(final, labels)
    loss7_2 = criterion_perceptual(mid_rgb, labels_rgb)

    total_loss = 1 * loss0 + 0.25 * loss1  \
                 + 0.2 * loss7 \
                 + 1 * loss0_2 + 0.25 * loss1_2 + 0.2 * loss7_2 \
                 + 1 * loss0_lab + 0.25 * loss1_lab # lab

    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss7, args['train_batch_size'], curr_iter, optimizer)

    return

def print_log(total_loss, loss0, loss1, loss2, batch_size, curr_iter, optimizer, type='normal'):
    total_loss_record.update(total_loss.data, batch_size)
    loss0_record.update(loss0.data, batch_size)
    loss1_record.update(loss1.data, batch_size)
    loss2_record.update(loss2.data, batch_size)

    log = '[iter %d][%s], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f] ' \
          '[lr %.13f]' % \
          (curr_iter, type, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
           optimizer.param_groups[0]['lr'])
    print(log)
    open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()
