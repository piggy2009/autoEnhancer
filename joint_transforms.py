import numbers
import random
from torchvision.transforms import functional as F
import torchvision
import cv2
import numpy as np

from PIL import Image, ImageOps

def flip(img, mask, flip_p):
    if flip_p < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def scale(img, mask, ratio):
    new_dims = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_dims, Image.BILINEAR), mask.resize(new_dims, Image.NEAREST)

def crop(img, mask, tw, th, x1, y1):

    return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

def rotate(img, mask, random, degree=10):
    rotate_degree = random * 2 * degree - degree
    return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        # assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Compose_single(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        # assert img.size == mask.size
        for t in self.transforms:
            img = t(img)
        return img

class ImageResize(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        if isinstance(img, list) and isinstance(mask, list):
            img_resized = []
            mask_resized = []
            for img_s, mask_s in zip(img, mask):
                tw, th = self.size
                img_resized.append(img_s.resize((tw, th), Image.BILINEAR))
                mask_resized.append(mask_s.resize((tw, th), Image.BILINEAR))
            return img_resized, mask_resized
        else:
            tw, th = self.size
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

class ImageResize_numpy(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if isinstance(img, list):
            img_resized = []
            for img_s in img:
                tw, th = self.size

                tmp = cv2.resize(img_s, (tw, th))
                if tmp.ndim == 2:
                    img_resized.append(tmp[:, :, np.newaxis])
                else:
                    img_resized.append(tmp)
                # mask_resized.append(mask_s.resize((tw, th), Image.BILINEAR))
            return img_resized
        else:
            tw, th = self.size
            return cv2.resize(img, (tw, th))


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if isinstance(img, list) and isinstance(mask, list):
            imgs_crop = []
            masks_crop = []
            img_first = True
            for img_s, mask_s in zip(img, mask):
                if self.padding > 0:
                    img_s = ImageOps.expand(img_s, border=self.padding, fill=0)
                    mask_s = ImageOps.expand(mask_s, border=self.padding, fill=0)

                assert img_s.size == mask_s.size
                w, h = img_s.size
                th, tw = self.size

                if w < tw or h < th:
                    imgs_crop.append(img_s.resize((tw, th), Image.BILINEAR))
                    masks_crop.append(mask_s.resize((tw, th), Image.BILINEAR))
                else:
                    if img_first:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        img_first = False
                    imgs_crop.append(img_s.crop((x1, y1, x1 + tw, y1 + th)))
                    masks_crop.append(mask_s.crop((x1, y1, x1 + tw, y1 + th)))
            return imgs_crop, masks_crop
        else:
            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=0)
                mask = ImageOps.expand(mask, border=self.padding, fill=0)

            assert img.size == mask.size
            w, h = img.size
            th, tw = self.size
            if w == tw and h == th:
                return img, mask
            if w < tw or h < th:
                return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.BILINEAR)

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomCrop_numpy(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if isinstance(img, list):
            imgs_crop = []
            img_first = True
            for img_s in img:

                # w, h, c = img_s.shape
                w = img_s.shape[0]
                h = img_s.shape[1]
                th, tw = self.size

                if w < tw or h < th:
                    tmp = cv2.resize(img_s, (tw, th))
                    if tmp.ndim == 2:
                        imgs_crop.append(tmp[:, :, np.newaxis])
                    else:
                        imgs_crop.append(tmp)

                else:
                    if img_first:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        img_first = False
                    imgs_crop.append(img_s[x1:x1 + tw, y1:y1 + th, :])
                    # masks_crop.append(mask_s.crop((x1, y1, x1 + tw, y1 + th)))
            return imgs_crop
        else:

            w, h = img.size
            th, tw = self.size
            if w == tw and h == th:
                return img
            if w < tw or h < th:
                return cv2.resize(img, (tw, th))

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img[x1:x1 + tw, y1:y1 + th, :]

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if isinstance(img, list) & isinstance(mask, list):
            if random.random() < 0.5:
                img_flips = []
                mask_flips = []
                for img_s, mask_s in zip(img, mask):
                    img_flips.append(img_s.transpose(Image.FLIP_LEFT_RIGHT))
                    mask_flips.append(mask_s.transpose(Image.FLIP_LEFT_RIGHT))
                return img_flips, mask_flips
            return img, mask
        else:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask

class RandomHorizontallyFlip_numpy(object):
    def __call__(self, img):
        if isinstance(img, list):
            if random.random() < 0.5:
                img_flips = []
                for img_s in img:
                    img_flips.append(np.flip(img_s, axis=-2))

                return img_flips
            return img
        else:
            if random.random() < 0.5:
                return np.flip(img, axis=-2)
            return img

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        if isinstance(img, list) & isinstance(mask, list):
            rotate_degree = random.random() * 2 * self.degree - self.degree
            img_rotates = []
            mask_rotates = []
            for img_s, mask_s in zip(img, mask):
                img_rotates.append(img_s.rotate(rotate_degree, Image.BILINEAR))
                mask_rotates.append(mask_s.rotate(rotate_degree, Image.NEAREST))
            return img_rotates, mask_rotates
        else:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        if isinstance(img, list):
            img_colorjitt = []
            img_colorjitt.append(transform(img[0]))
            img_colorjitt.append(img[1])
            return img_colorjitt, mask
        else:
            return transform(img), mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
