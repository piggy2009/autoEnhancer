import os
import numpy as np
from PIL import Image
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure


from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import peak_signal_noise_ratio as psnr1


ckpt_path = '/home/ty/code/NewIdeaTest/ckpt/WaterEnhance_2022-03-29 02:12:01/136000/EUVP'
gt_path = '/home/ty/data/EUVP/test_samples/GTr'

psnr_record = AvgMeter()
ssim_record = AvgMeter()
results = {}

images = os.listdir(ckpt_path)
images.sort()
image_names = []
psnr_list = []
for name in images:
    img = Image.open(os.path.join(ckpt_path, name)).convert('RGB')
    img = np.array(img)

    gt = Image.open(os.path.join(gt_path, name[:-4] + '.jpg')).convert('RGB')
    gt = np.array(gt)
    psnr = psnr1(img, gt)
    ssim = ssim1(img, gt, multichannel=True)
    psnr_record.update(psnr)
    ssim_record.update(ssim)
    # each = {'name': name, 'psnr': psnr}
    image_names.append(name)
    psnr_list.append(psnr)

results['UIEB'] = {'PSNR': psnr_record.avg, 'SSIM': ssim_record.avg}
print(results)



