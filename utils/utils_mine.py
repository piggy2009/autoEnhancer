import torch
# from models.net import SNet
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import sys
sys.path.append('/home/tangyi/code/SVS')
plt.style.use('classic')

def load_part_of_model(new_model, src_model_path, device_id=0):
    src_model = torch.load(src_model_path, map_location='cuda:' + str(device_id))
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model2(new_model, src_model_path, device_id=0):
    src_model = torch.load(src_model_path, map_location='cuda:' + str(device_id))
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print(k)
        param = src_model.get(k)
        k = k.replace('module.', '')
        m_dict[k].data = param

    new_model.load_state_dict(m_dict)
    return new_model

def visualize(input, save_path):
    input = input.data.cpu().numpy()
    for i in range(input.shape[1]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(input[0, i, :, :])

    plt.savefig(save_path)

def visualize_vec(input_vec, save_path):
    input = input_vec.data.cpu().numpy()
    input = np.squeeze(input)
    input = np.tile(input, (64, 1))
    plt.imshow(input)
    plt.colorbar()
    plt.savefig(save_path)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    ckpt_path = './ckpt'
    exp_name = 'VideoSaliency_2019-05-14 17:13:16'

    args = {
        'snapshot': '30000',  # your snapshot filename (exclude extension name)
        'crf_refine': False,  # whether to use crf to refine results
        'save_results': True,  # whether to save the resulting masks
        'input_size': (473, 473)
    }
    a = torch.rand([1, 64, 1, 1])
    # visualize_vec(a, 'a.png')
    # from MGA.mga_model import MGA_Network
    # a = MGA_Network(nInputChannels=3, n_classes=1, os=16,
    #             img_backbone_type='resnet101', flow_backbone_type='resnet34')
    # load_MGA(a, '../pre-trained/MGA_trained.pth')

    # net = SNet(cfg=None).cuda()
    # net = fuse_MGA_F3Net('../pre-trained/MGA_trained.pth', '../pre-trained/F3Net.pth', net)
    # torch.save(net.state_dict(), '../pre-trained/SNet.pth')
    # net = load_part_of_model(net, '../pre-trained/SNet.pth')
    # input = torch.zeros([2, 3, 380, 380]).cuda()
    # output = net(input, input)
    # print(output[0].size())
    # src_model_path = os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
    # net = R3Net(motion='GRU')
    # net = load_part_of_model(net, src_model_path)
