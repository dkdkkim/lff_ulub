from typing import List
import numpy as np
import json, os, pickle

save_dir = 'dataset_ulub/CelebA-HQ'
if not os.path.exists(os.path.join(save_dir, 'train')):
    os.makedirs(os.path.join(save_dir, 'train'))
if not os.path.exists(os.path.join(save_dir, 'valid_ub1')):
    os.makedirs(os.path.join(save_dir, 'valid_ub1'))
if not os.path.exists(os.path.join(save_dir, 'valid_ub2')):
    os.makedirs(os.path.join(save_dir, 'valid_ub2'))
attr_names = ['gender']
with open(os.path.join(save_dir, 'attr_names.pkl'), 'wb') as file:
    pickle.dump(attr_names, file)

''' train '''
txt_file = 'dataset_ulub/CelebA-HQ/train.txt' # n of train : 22537
with open(txt_file) as txt_file:
    image_list: List[str] = txt_file.read().splitlines()

imgs, lbls = [], []
for image in image_list:
    img_path, label = image.split(',')
    imgs.append(img_path)
    lbls.append(int(label))

attrs = np.array(lbls)

np.save(os.path.join(save_dir, 'train/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'train/images.json'), 'w') as file:
    json.dump(imgs, file)

''' valid ub1'''
txt_file = 'dataset_ulub/CelebA-HQ/ub1_val.txt'
with open(txt_file) as txt_file:
    image_list: List[str] = txt_file.read().splitlines()

imgs, lbls = [], []
for image in image_list:
    img_path, label = image.split(',')
    imgs.append(img_path)
    lbls.append(int(label))

attrs = np.array(lbls)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'valid_ub1/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'valid_ub1/images.json'), 'w') as file:
    json.dump(imgs, file)

''' valid ub2'''
txt_file = 'dataset_ulub/CelebA-HQ/ub2_val.txt'
with open(txt_file) as txt_file:
    image_list: List[str] = txt_file.read().splitlines()

imgs, lbls = [], []
for image in image_list:
    img_path, label = image.split(',')
    imgs.append(img_path)
    lbls.append(int(label))

attrs = np.array(lbls)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'valid_ub2/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'valid_ub2/images.json'), 'w') as file:
    json.dump(imgs, file)






