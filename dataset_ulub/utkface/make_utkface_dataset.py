import json, os, pickle
from pathlib import Path
import numpy as np

def get_skintone_class(skintone):
    if skintone == 0:
        skintone_class = 0
    else:
        skintone_class = 1
    return skintone_class

json_dir = 'dataset_ulub/utkface'
data_path = Path('/data/dk/ulub/dataset/utkface/utkcropped')
var = 0.2
data = []

save_dir = 'dataset_ulub/utkface'
if not os.path.exists(os.path.join(save_dir, 'test')):
    os.makedirs(os.path.join(save_dir, 'test'))
if not os.path.exists(os.path.join(save_dir, 'ub1')):
    os.makedirs(os.path.join(save_dir, 'ub1'))
if not os.path.exists(os.path.join(save_dir, 'ub2')):
    os.makedirs(os.path.join(save_dir, 'ub2'))
attr_names = ['skintone', 'gender']
with open(os.path.join(save_dir, 'attr_names.pkl'), 'wb') as file:
    pickle.dump(attr_names, file)


with open(os.path.join(json_dir, 'white_female.json'), 'r') as file:
    wf = json.load(file)
with open(os.path.join(json_dir, 'white_male.json'), 'r') as file:
    wm = json.load(file)
with open(os.path.join(json_dir, 'black_female.json'), 'r') as file:
    bf = json.load(file)
with open(os.path.join(json_dir, 'black_male.json'), 'r') as file:
    bm = json.load(file)
with open(os.path.join(json_dir, 'asian_female.json'), 'r') as file:
    af = json.load(file)
with open(os.path.join(json_dir, 'asian_male.json'), 'r') as file:
    am = json.load(file)
with open(os.path.join(json_dir, 'indian_female.json'), 'r') as file:
    indf = json.load(file)
with open(os.path.join(json_dir, 'indian_male.json'), 'r') as file:
    indm = json.load(file)

wf_len, wm_len, bf_len, bm_len, af_len, am_len, indf_len, indm_len \
    = len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(indf), len(indm)
# print(len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(indf), len(indm))
wf_back, wf_data = wf[:int(var * wm_len)], wf[int(var * wm_len):]
wm_back, wm_data = wm[:int(var * wf_len)], wm[int(var * wf_len):]
bf_back, bf_data = bf[:int(var * bm_len)], bf[int(var * bm_len):]
bm_back, bm_data = bm[:int(var * bf_len)], bm[int(var * bf_len):]
af_back, af_data = af[:int(var * am_len)], af[int(var * am_len):]
am_back, am_data = am[:int(var * af_len)], am[int(var * af_len):]
indf_back, indf_data = indf[:int(var * indm_len)], indf[int(var * indm_len):]
indm_back, indm_data = indm[:int(var * indf_len)], indm[int(var * indf_len):]

# for a in [wf_back, wf_data, wm_back, wm_data, bf_back, bf_data, bm_back, bm_data, af_back, af_data, am_back, am_data, indf_back, indf_data, indm_back, indm_data]:
#     print(len(a))
# input('enter')

''' ub1(9629) '''
img_list = wf_data + wm_back[:int(len(wf_data) * var)] + \
            bm_data + bf_back[:int(len(bm_data) * var)] + \
            am_data + af_back[:int(len(am_data) * var)] + \
            indm_data + indf_back[:int(len(indm_data) * var)]

imgs, lbls = [], []

for img in img_list:
    imgs.append(str(data_path / img))
    lbls.append([get_skintone_class(int(img.split('_')[2])), int(img.split('_')[1])])
attrs = np.array(lbls)

np.save(os.path.join(save_dir, 'ub1/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'ub1/images.json'), 'w') as file:
    json.dump(imgs, file)
    # print(len(wf_data), len(wm_back[:int(len(wf_data) * var)]))
    # print(len(bm_data)+len(am_data)+len(indm_data), len(bf_back[:int(len(bm_data) * var)])+len(af_back[:int(len(am_data) * var)])+len(indf_back[:int(len(indm_data) * var)]))

''' ub2(10352) '''
img_list = wm_data + wf_back[:int(len(wm_data) * var)] + \
            bf_data + bm_back[:int(len(bf_data) * var)] + \
            af_data + am_back[:int(len(af_data) * var)] + \
            indf_data + indm_back[:int(len(indf_data) * var)]
    # print(len(wm_data), len(wf_back[:int(len(wm_data) * var)]))
    # print(len(bf_data) + len(af_data) + len(indf_data),
    #       len(bm_back[:int(len(bf_data) * var)]) + len(am_back[:int(len(af_data) * var)]) + len(
    #           indm_back[:int(len(indf_data) * var)]))
imgs, lbls = [], []

for img in img_list:
    imgs.append(str(data_path / img))
    lbls.append([get_skintone_class(int(img.split('_')[2])), int(img.split('_')[1])])
attrs = np.array(lbls)

np.save(os.path.join(save_dir, 'ub2/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'ub2/images.json'), 'w') as file:
    json.dump(imgs, file)

''' test(120) '''
with open(os.path.join(json_dir, 'skintone_gender_test.json'), 'r') as file:
    img_list = json.load(file)
imgs, lbls = [], []

for img in img_list:
    imgs.append(str(data_path / img))
    lbls.append([get_skintone_class(int(img.split('_')[2])), int(img.split('_')[1])])
attrs = np.array(lbls)

np.save(os.path.join(save_dir, 'test/attrs.npy'), attrs)
with open(os.path.join(save_dir, 'test/images.json'), 'w') as file:
    json.dump(imgs, file)