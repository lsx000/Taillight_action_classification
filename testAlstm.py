# -*- coding: utf-8 -*-
"""
@author: lsx
"""
#from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np 
import cv2 
import torch.nn.functional as func
import iterater as ite
import math
import matplotlib.pyplot as plt
import os
from ALSTM import ALSTM,lstm_cell
from model import SegNet


def reload_net():
    
    trainednet = torch.load('./model_cfair10_2.pth')
    
    return trainednet

def test():
    loader = ite.data_test()
    
    model = reload_net()
    
    print (model)
    
    
    
    tars = [0]*5
    pres = [0]*5
    
    scores = []
    las = []
    lls = []
    y_true = []
    
    for i,data in enumerate(loader):
        x,y,fn= data
        
        y_pred,l = model(x)
        label = y[:,-1,:]
        pred = y_pred[:,-1,:]
        
        for j in range(x.size(0)):
            
            ta = torch.argmax(label[j]).item()
            pr = torch.argmax(pred[j]).item()
            y_true.append([ta,pr])            
            if pr == 5:
                pr = 0
            
            tars[ta] +=1
            if ta == pr:
                
                pres[pr] += 1
            lls.append(ta)
            scores.append(pred[j].cpu().detach().numpy())
            las.append(label[j].cpu().detach().numpy())
            

    
       
    recall = 0    
    for i in range(5):       
        recall += 1.*pres[i]/tars[i]
    recall = recall/5.
    print(recall)
        
    
    

    
            
        
        
if __name__ == '__main__':
    test()
    
        
