# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:17 2019

@author: ZeyuGao
"""

import os
import numpy as np
import cv2
import random
from tqdm import tqdm
import pandas as pd

def get_all_files(path):
    # 读取所有文件路径
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_list.append(os.path.join(dirpath,filename).replace('\\', '/'))
    return file_list

def split_train_test(wsi_list_path, train_num=40):
    for root, dirs, files in os.walk("/home1/gzy/Subtype/MetaData/2000"):
        folder_list = dirs
        if folder_list:
            break
    label_df = pd.read_csv(wsi_list_path)
    classes = label_df['label'].max()+1
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for i in range(classes):
        train_index = []
        test_index = []
        temp_df = label_df[label_df['label'] == i]
        temp_df = temp_df.sample(frac=1)
        temp_df.reset_index(drop=True, inplace=True)
        for j in range(temp_df.shape[0]):
            if '.'.join(temp_df.loc[j,'filename'].split('.')[:-1]) in folder_list and len(train_index)<train_num:
                train_index.append(j)
            else:
                test_index.append(j)
        if train_set.empty:
            train_set = temp_df.iloc[train_index]
        else:
            train_set = pd.concat([train_set, temp_df.iloc[train_index]], axis=0)
        if test_set.empty:
            test_set = temp_df.iloc[test_index]
        else:
            test_set = pd.concat([test_set, temp_df.iloc[test_index]], axis=0)
    train_set.to_csv("/home1/gzy/Subtype/train_labels.txt", index=False)
    test_set.to_csv("/home1/gzy/Subtype/test_labels.txt", index=False)

def get_mean_std(imgs_path):
    class CalMeanVar():
        def __init__(self):
            self.count = 0
            self.A = 0
            self.A_ = 0
            self.V = 0
    
        def cal(self, data):
            self.count += 1
            if self.count == 1:
                self.A_ = data
                self.A = data
                return
            self.A_ = self.A
            self.A = self.A + (data - self.A) / self.count
            self.V = (self.count - 1) / self.count ** 2 * (data - self.A_)**2 + (self.count - 1)/self.count * self.V
    
    img_h, img_w = 64, 64   #根据自己数据集适当调整，影响不大
    num_C = 100000
    imgs_path_list = get_all_files(imgs_path)
    random.shuffle(imgs_path_list)
    B=CalMeanVar()
    G=CalMeanVar()
    R=CalMeanVar()
    for item in tqdm(imgs_path_list[:num_C]):
        img = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(img_w,img_h))
        #img = img/255
        b_list = img[:,:,0].ravel()
        g_list = img[:,:,1].ravel()
        r_list = img[:,:,2].ravel()
        for i in range(len(b_list)):
            B.cal(b_list[i])
            G.cal(g_list[i])
            R.cal(r_list[i])
    
    print("normMean = {}".format([R.A, G.A, B.A]))
    print("normStd = {}".format([np.sqrt(R.V),np.sqrt(G.V),np.sqrt(B.V)]))
    
if __name__ == '__main__':
    #wsi_list_path = "/home1/gzy/Subtype/labels.txt"
    #split_train_test(wsi_list_path, train_num=40)
    img_path = '/home1/may/hover_data/train/images/'
    get_mean_std(img_path)
