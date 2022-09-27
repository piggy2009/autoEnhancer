# coding: utf-8
import os

# szu 169 sever

datasets_root = '/home/ty/data'

save_root = './'

# local pc
# datasets_root = '/home/qub/data/saliency'

saving_path = os.path.join(save_root, 'ckpt')

# For each dataset, I put images and masks together
msra10k_path = os.path.join(datasets_root, 'msra10k')
ecssd_path = os.path.join(datasets_root, 'ecssd')
hkuis_path = os.path.join(datasets_root, 'hkuis')
pascals_path = os.path.join(datasets_root, 'pascals')
dutomron_path = os.path.join(datasets_root, 'dutomron')
sod_path = os.path.join(datasets_root, 'sod')
video_train_path = os.path.join(datasets_root, 'Pre-train')
davis_path = os.path.join(datasets_root, 'davis')
fbms_path = os.path.join(datasets_root, 'FBMS')
mcl_path = os.path.join(datasets_root, 'MCL')
uvsd_path = os.path.join(datasets_root, 'UVSD')
visal_path = os.path.join(datasets_root, 'ViSal')
vos_path = os.path.join(datasets_root, 'VOS')
segtrack_path = os.path.join(datasets_root, 'SegTrack-V2')
davsod_path = os.path.join(datasets_root, 'DAVSOD')
video_seq_path = os.path.join(datasets_root, 'video_saliency/train_all')
video_seq_gt_path = os.path.join(datasets_root, 'video_saliency/train_all_gt2_revised')
