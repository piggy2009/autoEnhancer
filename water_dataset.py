import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image, ImageCms
from matplotlib import pyplot as plt
import cv2
import random
import torchvision
import numpy as np
from joint_transforms import crop, scale, flip, rotate
from torchvision import transforms

def convert_from_image_to_cv2(img):
    return np.array(img)

def convert_from_BGR_to_RGB(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def convert_from_image_to_hsv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

def convert_from_image_to_lab(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)

class WaterImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = os.listdir(os.path.join(root, 'input_train'))
        self.imgs.sort()
        self.depths = os.listdir(os.path.join(root, 'depth_train'))
        self.depths.sort()
        self.labels = os.listdir(os.path.join(root, 'gt_train'))
        self.labels.sort()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'input_train', self.imgs[index])).convert('RGB')
        target = Image.open(os.path.join(self.root, 'gt_train', self.labels[index])).convert('RGB')
        depth = Image.open(os.path.join(self.root, 'depth_train', self.depths[index])).convert('L')
        img_list = []
        gt_list = []
        img_list.append(img)
        img_list.append(target)
        gt_list.append(depth)
        gt_list.append(depth)
        if self.joint_transform is not None:
            img_list, gt_list = self.joint_transform(img_list, gt_list)

        img = img_list[0]
        # hsv = img_list[0].convert('HSV')
        hsv = convert_from_image_to_hsv(img_list[0])
        # lab = img_list[0].convert('HSV')
        lab = convert_from_image_to_lab(img_list[0])
        lab_target = convert_from_image_to_lab(img_list[1])
        target = convert_from_image_to_cv2(img_list[1])
        if self.transform is not None:
            img = self.transform(img)
            hsv = self.transform(hsv)
            lab = self.transform(lab)
            target = self.transform(target)
            lab_target = self.transform(lab_target)
        if self.target_transform is not None:
            depth = self.target_transform(gt_list[0])

        return img, hsv, lab, target, lab_target, depth

    def __len__(self):
        return len(self.imgs)

class WaterImage2Folder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = os.listdir(os.path.join(root, 'train_input'))
        self.imgs.sort()
        self.labels = os.listdir(os.path.join(root, 'train_gt'))
        self.labels.sort()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'train_input', self.imgs[index])).convert('RGB')
        target = Image.open(os.path.join(self.root, 'train_gt', self.labels[index])).convert('RGB')
        # print(self.imgs[index], '--', self.labels[index])

        img_list = []
        gt_list = []
        img_list.append(img)
        img_list.append(target)
        gt_list.append(img.convert('L'))
        gt_list.append(img.convert('L'))
        if self.joint_transform is not None:
            img_list, gt_list = self.joint_transform(img_list, gt_list)

        img = img_list[0]
        # hsv = img_list[0].convert('HSV')
        hsv = convert_from_image_to_hsv(img_list[0])
        # lab = img_list[0].convert('HSV')
        lab = convert_from_image_to_lab(img_list[0])
        lab_target = convert_from_image_to_lab(img_list[1])
        target = convert_from_image_to_cv2(img_list[1])
        if self.transform is not None:
            img = self.transform(img)
            hsv = self.transform(hsv)
            lab = self.transform(lab)
            target = self.transform(target)
            lab_target = self.transform(lab_target)

        return img, hsv, lab, target, lab_target

    def __len__(self):
        return len(self.imgs)

class WaterImage3Folder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = os.listdir(os.path.join(root, 'input_train_uw'))
        self.imgs.sort()
        self.labels = os.listdir(os.path.join(root, 'gt_train_uw'))
        self.labels.sort()
        self.segments = os.listdir(os.path.join(root, 'segment_train_uw', 'FV'))
        self.labels.sort()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root, 'input_train_uw', self.imgs[index]))
        target = cv2.imread(os.path.join(self.root, 'gt_train_uw', self.labels[index]))

        fv = cv2.imread(os.path.join(self.root, 'segment_train_uw', 'FV', self.segments[index]), 0)
        hd = cv2.imread(os.path.join(self.root, 'segment_train_uw', 'HD', self.segments[index]), 0)
        ri = cv2.imread(os.path.join(self.root, 'segment_train_uw', 'RI', self.segments[index]), 0)
        ro = cv2.imread(os.path.join(self.root, 'segment_train_uw', 'RO', self.segments[index]), 0)
        wr = cv2.imread(os.path.join(self.root, 'segment_train_uw', 'WR', self.segments[index]), 0)
        # fv = cv2.resize(fv, (224, 224))
        # print(fv.shape, '-', hd.shape)

        # segmentation = np.stack((fv, hd, ri, ro, wr), axis=0)
        img_list = []
        img_list.append(img)
        img_list.append(target)
        img_list.append(fv[:, :, np.newaxis])
        img_list.append(hd[:, :, np.newaxis])
        img_list.append(ri[:, :, np.newaxis])
        img_list.append(ro[:, :, np.newaxis])
        img_list.append(wr[:, :, np.newaxis])

        if self.joint_transform is not None:
            img_list = self.joint_transform(img_list)

        img = convert_from_BGR_to_RGB(img_list[0])
        # hsv = img_list[0].convert('HSV')
        hsv = convert_from_image_to_hsv(img)
        # lab = img_list[0].convert('HSV')
        lab = convert_from_image_to_lab(img)
        target = convert_from_BGR_to_RGB(img_list[1])
        lab_target = convert_from_image_to_lab(target)
        segmentation = np.concatenate((img_list[2], img_list[3], img_list[4], img_list[5], img_list[6]), axis=-1)
        # print(img.shape, '-', lab.shape, '-', target.shape, '-', segmentation.shape)
        if self.transform is not None:
            img = self.transform(img)
            hsv = self.transform(hsv)
            lab = self.transform(lab)
            target = self.transform(target)
            lab_target = self.transform(lab_target)
            segmentation = self.transform(segmentation)

        # L = lab[[0], ...] / 50. - 1.  # Between -1 and 1
        # ab = lab_target[[1, 2], ...] / 110.  # Between -1 and 1

        return img, hsv, lab, target, lab_target, segmentation

    def __len__(self):
        return len(self.imgs)

class WaterImage4Folder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, input_size):
        self.root = root
        self.imgs = os.listdir(root)
        self.imgs.sort()
        self.transforms = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),  # A little data augmentation!
        ])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.imgs[index])).convert("RGB")

        img = self.transforms(img)
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab = transforms.ToTensor()(lab)
        img = transforms.ToTensor()(img)
        # L = lab[[0], ...] / 50. - 1.  # Between -1 and 1
        # ab = lab[[1, 2], ...] / 110.  # Between -1 and 1


        return img, lab

    def __len__(self):
        return len(self.imgs)

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}

    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

if __name__ == '__main__':
    from torchvision import transforms, models

    import joint_transforms
    from torch.utils.data import DataLoader
    joint_transform = joint_transforms.Compose([
        # joint_transforms.ImageResize(256),
        joint_transforms.RandomCrop(256),
        joint_transforms.RandomHorizontallyFlip(),
    ])

    joint_transform2 = joint_transforms.Compose([
        # joint_transforms.ImageResize(256),
        joint_transforms.RandomCrop(128),
        joint_transforms.RandomHorizontallyFlip(),
    ])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.ToTensor()
    input_size = (200, 200)

    train_set2 = WaterImage2Folder('/Users/tangyi/mycode/5k/eval',
                                  joint_transform, img_transform, target_transform)

    train_loader = DataLoader(train_set2, batch_size=2, num_workers=1, shuffle=False)
    # train_set3 = WaterImage2Folder('/Users/tangyi/mycode/5k/eval',
    #                                joint_transform2, img_transform, target_transform)
    #
    # train_loader2 = DataLoader(train_set3, batch_size=4, num_workers=1, shuffle=False)
    # train_loader2 = DataLoader(train_set2, batch_size=6, num_workers=4, shuffle=True)
    # dataloader_iterator = iter(train_loader2)
    # vgg = models.vgg19(pretrained=True).features
    # for param in vgg.parameters():
    #     param.requires_grad_(False)
    # vgg.eval()
    for i, data in enumerate(train_loader):
        print('i=', i)
        # data1, data2 = data
        # inputs, flows, labels, inputs2, labels2 = data
        # data2 = next(dataloader_iterator)
        rgb, hsv, lab, target, lab_target = data
        print(rgb.shape)
        # texture_features = get_features(rgb, vgg)
        # target_features = get_features(target, vgg)
        #
        # content_loss = torch.mean((texture_features['conv4_2'] - target_features['conv4_2']) ** 2)

        # first, second = data
        # rgb = rgb.data.cpu().numpy()
        # hsv = hsv.data.cpu().numpy()
        # depth = depth.data.cpu().numpy()
        # target = target.data.cpu().numpy()
        # # target = target2.data.cpu().numpy()
        # # # np.savetxt('image.txt', input[0, 0, :, :])
        # rgb = rgb.transpose(0, 2, 3, 1)
        # hsv = hsv.transpose(0, 2, 3, 1)
        # depth = depth.transpose(0, 2, 3, 1)
        # # flow = flow.transpose(0, 2, 3, 1)
        # target = target.transpose(0, 2, 3, 1)
        # # # # for i in range(0, input.shape[0]):
        # plt.subplot(2, 2, 1)
        # plt.imshow(rgb[0, :, :, :])
        # plt.subplot(2, 2, 2)
        # plt.imshow(target[0, :, :, :])
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(hsv[0, :, :, :])
        # plt.subplot(2, 2, 4)
        # plt.imshow(depth[0, :, :, 0])
        #
        # plt.show()
        # print(lab.size())
        # print(depth.size())

    # for j, data in enumerate(train_loader2):
    #     print('j=', j)
    #     rgb, hsv, lab, target, lab_target = data
    #     print(rgb.shape)
