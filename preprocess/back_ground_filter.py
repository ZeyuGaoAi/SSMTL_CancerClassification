# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:03:45 2019

@author: ZeyuGao
"""

import cv2
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from utils import get_all_files

def generate_binary_mask_for_wsi(file_path, output_folder):
    # 使用CV2的OTSU进行二值化
    oslide = openslide.OpenSlide(file_path)
    magnification = oslide.properties.get('aperio.AppMag')
#    width = oslide.dimensions[0]
#    height = oslide.dimensions[1]
    level = oslide.level_count - 1
    if level > 3:
       level = 3
#    scale_down = oslide.level_downsamples[level]
    w, h = oslide.level_dimensions[level]
    # 防止出现没有放大倍数直接处理原图的情况
    if level < 1:
        print(file_path)
        oslide.close()
        return
        patch = oslide.read_region((0, 0), 0, (w, h))
        patch = patch.resize((int(w/32), int(h/32)), Image.ANTIALIAS)
    else:
        patch = oslide.read_region((0, 0), level, (w, h))
    slide_id = file_path.split('/')[-1].split('.svs')[0]
    patch.save('{}/{}_resized.png'.format(output_folder, slide_id));
    img = cv2.cvtColor(np.asarray(patch),cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (61, 61), 0)
    ret, img_filtered = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fname = '{}/{}_mask.png'.format(output_folder, slide_id)
    cv2.imwrite(fname, img_filtered)
    oslide.close()
    return magnification
    
if __name__ == '__main__':
    input_folder = "/home5/svs_files/"
    mask_folder = "/home1/gzy/Gastric_Subtype/WSIMask"
    file_list = get_all_files(input_folder)
    file_list = [file for file in file_list if file.split('.')[-1] == 'svs']
    wsi_list_path = "/home1/gzy/Gastric_Subtype/Stomach_slide_table.txt"
    wsi_df = pd.read_csv(wsi_list_path)
    wsi_list = wsi_df['filename'].tolist()
    file_list = [file for file in file_list if file.split('/')[-1] in wsi_list]
    mag = []
    for file in tqdm(file_list):
        if '.svs' in file:
            try:
                mag.append(generate_binary_mask_for_wsi(file, mask_folder))
            except Exception as e:
                print('%s: %s' % (file, e))

    print(mag)
