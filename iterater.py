# -*- coding: utf-8 -*-
"""
@author: lsx
"""
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms, utils


import random
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


    
def tail_pic_read(path_img):
    
    channel = 3
    time = 16
    h = 224
    w = 224
    x = torch.zeros(time,channel,h,w)
    y = torch.zeros(time,5)
    data = pd.read_csv(path_img+"label.csv",encoding = 'utf-16')
    # print("DATA",data,path_img)
    # print(data.values)
    da = []
    for line in data.values:
        # print("line",line)
        da.append(line)
    # print(da)
    label = int(da[0])
    
    
    if label < 6:
        for j in range(time):
            y[j][label] = 1.
    else:
        for j in range(time):
            y[j][0] = 1.
    for i in range(16):
        
        frame = cv2.imread(path_img+str(i)+'.png')
        #cv2.imshow('image', frame)
        
        frame = cv2.resize(frame,(h,w))
        blur=cv2.GaussianBlur(frame,(5,5),0)
        frame = blur/255.0
        tensor_cv = torch.from_numpy(np.transpose(frame, (2, 0, 1))).cuda()
        #tensor_cv = tensor_cv/255.0
        #print tensor_cv
        x[i,:,:,:]= tensor_cv
        
        #cv2.waitKey(0)
    print (path_img)
    return x.cuda(),y.cuda()




class trainset(Dataset):
    def __init__(self,path,loader=tail_pic_read):
        
        self.images = path        
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img,target = self.loader(fn+'/')
        #target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
     
def dataiter():
    
    path = '/traintail'
    name_list = os.listdir(path)
    path_list = []
    for name in name_list:
        path_list.append(path+name)
    train_data  = trainset(path = path_list)
    loader = DataLoader(train_data, batch_size=4,shuffle=True)
    data_iter = iter(loader)
    return data_iter
class testset(Dataset):
    def __init__(self,path,loader=tail_pic_read):
        
        self.images = path
        
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img,target = self.loader(fn+'/')
        
        return img,target,fn

    def __len__(self):
        return len(self.images)
def data_test():
    
    path = '/testtail'
    
    name_list = os.listdir(path)
    name_list.sort()
    path_list = []
    for name in name_list:
        path_list.append(path+name)
    train_data  = testset(path = path_list)
    loader = DataLoader(train_data, batch_size=1,shuffle=False)
    data_iter = iter(loader)
    return data_iter
    
if __name__ == '__main__':
    
    for i in range(2):
        data_iter =dataiter()
        x,label = data_iter.next()
        print (x)
        '''
        for j in range(x.size(1)):
            image = x[0,j,:,:,:].cpu().numpy()
            img = np.array(image,dtype=np.uint8)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img*255.)
            plt.show()
        '''
        
    
    
    
    
    
    
    
    
    
