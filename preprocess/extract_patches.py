# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:17 2019

@author: ZeyuGao
"""

import os
import cv2
import openslide
from tqdm import tqdm
from PIL import Image
import pandas as pd

from utils import get_all_files
    
def cut_patches_from_wsi(file_path, output_folder, mask_folder,
                         size=1000, step=500, rate=0.8, output_size=512):
    # 将wsi划窗切分成指定大小的patches
    slide_id = file_path.split('/')[-1].split('.svs')[0]
    mask_path = '{}/{}_mask.png'.format(mask_folder, slide_id)
    if not os.path.exists(mask_path):
        print("No mask file for this WSI!")
        return
    output_folder = os.path.join(output_folder, slide_id)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # 处理wsi
    oslide = openslide.OpenSlide(file_path)
    magnification = oslide.properties.get('aperio.AppMag')
    if magnification is None:
        print("No magnification infos")
        return
    elif magnification == '20':
        size = int(size/2)
        step = int(step/2)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    level = oslide.level_count - 1
    if level > 3:
       level = 3
    w, h = oslide.level_dimensions[level]   
    mag_w = width/w
    mag_h = height/h
    mag_size = size/mag_w
    # 读取mask图
    mask = cv2.imread('{}/{}_mask.png'.format(mask_folder, slide_id), 0)
    mask = mask.T
    if not mask.shape == (w, h):
        print("Mask file not match for this WSI!")
        return
    corrs = []
    for x in range(1, width, step):
        for y in range(1, height, step):
            if x + size > width:
                continue
            else:
                w_x = size
            if y + size > height:
                continue
            else:
                w_y = size
            # 根据mask进行过滤，大于rate个背景则不要
            mask_patch = mask[int(x/mag_w):int(x/mag_w + mag_size),int(y/mag_h):int(y/mag_h + mag_size)]
            num = mask_patch[(mask_patch == 255)].size
            if num < mask_patch.size * rate:
                corrs.append((x, y, w_x, w_y))
    print(file_path, len(corrs))
    # wsi = oslide.read_region((0, 0), 0, (width, height))
    for corr in tqdm(corrs):
        x, y, w_x, h_y = corr
        # patch = wsi.crop((x, y, x+w_x, y+h_y))
        patch = oslide.read_region((x, y), 0, (w_x, h_y))
        patch = patch.resize((output_size, output_size), Image.ANTIALIAS)
        fname = '{}/{}_{}_{}.png'.format(output_folder, x, y, size)
        patch.save(fname)
    oslide.close()

if __name__ == '__main__':
    input_folder = "/home5/svs_files"
    mask_folder = "/home1/gzy/Lung_Subtype/WSIMask"
    output_folder = "/home5/gzy/LungDataset/"
    wsi_list_path = "/home1/gzy/Lung_Subtype/lung_slide_table.txt"
    wsi_df = pd.read_csv(wsi_list_path)
    wsi_list = wsi_df['filename'].tolist()
    file_list = get_all_files(input_folder)
    file_list = [file for file in file_list if file.split('.')[-1] == 'svs']
    file_list = [file for file in file_list if file.split('/')[-1] in wsi_list]
    for index in tqdm(range(0,len(file_list))):
        file_path = file_list[index]
        cut_patches_from_wsi(file_path, output_folder, mask_folder,
                             size=2000, step=2000, rate=0.6, output_size=512)
