#coding:utf-8

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np 
import cv2 
import torch.nn.functional as func
import iterater as ite
import math
from model import SegNet
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class lstm_cell(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(lstm_cell, self).__init__()

        self.input_num = input_num
        self.hidden_num = hidden_num

        self.Wxi = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whi = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxf = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whf = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxc = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whc = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxo = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Who = nn.Linear(self.hidden_num, self.hidden_num, bias=False)

    def forward(self, xt, ht_1, ct_1):        
        it = torch.sigmoid(self.Wxi(xt) + self.Whi(ht_1))        
        ft = torch.sigmoid(self.Wxf(xt) + self.Whf(ht_1))        
        ot = torch.sigmoid(self.Wxo(xt) + self.Who(ht_1))        
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(xt) + self.Whc(ht_1))        
        ht = ot * torch.tanh(ct)
        
        return  ht, ct

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))
def my_cross_loss(l_t_i, y_t_i, ypre_t_i,lam):
    N = ypre_t_i.size(0)
    for i in range(5):
        for j in range(16):
            for k in range(N):
                if ypre_t_i[k][j][i] < 1e-10:
                    ypre_t_i[k][j][i] = 1e-10
    f1 = -1*torch.sum( torch.sum( torch.mul(y_t_i,torch.log(ypre_t_i)),2),1)    
    T_l = torch.sum(l_t_i,1) 
    f2 = lam*torch.sum((1-T_l)**2,1)
    loss = f1+f2
    N = ypre_t_i.size(0)
    
    loss = torch.mean(loss)
    output = ypre_t_i[:,-1,:]
    prediction = torch.argmax(output, 1)
    print (prediction)
    label = torch.argmax(y_t_i[:,-1,:],1)
    print (label)
    
    for e,la in enumerate(label):
        if la.item()==3 or la.item()==4:
            loss*=2          
    
    acc = 0
    for e in range(N):
        if int(label[e].item())==int(prediction[e].item()):
            acc = acc+1
    acc = acc/(N*1.0)
    return loss,acc



class ALSTM(nn.Module):

    def __init__(self, input_num, hidden_num, num_layers,out_num ):
        
        super(ALSTM, self).__init__()

        # Make sure that `hidden_num` are lists having len == num_layers
        hidden_num = self._extend_for_multilayer(hidden_num, num_layers)
        if not len(hidden_num) == num_layers:
            raise ValueError('The length of hidden_num is not consistent with num_layers.')

        self.input_num = input_num
        self.hidden_num = hidden_num
        self.num_layers = num_layers
        self.out_num = out_num

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_num = self.input_num if i == 0 else self.hidden_num[i - 1]            
            cell_list.append(lstm_cell(cur_input_num,self.hidden_num[i]).cuda())           

        self.cell_list = nn.ModuleList(cell_list)
        #self.conv=nn.Sequential(*list(torchvision.models.resnet101(pretrained=True).children())[:-2])
        vgg = torchvision.models.vgg16(pretrained=True)
        self.conv=nn.Sequential(*list(vgg.features._modules.values())[:31])
        
        
            
        self.Wha=nn.Linear(self.hidden_num[-1],49)
        self.fc=nn.Linear(self.hidden_num[-1],self.out_num)
        self.softmax=nn.Softmax(dim=1)
        self.tanh=nn.Tanh()
        self.soft_out = nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(p=0.5)
        
    def forward(self, x, hidden_state=None):
        
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=x.size(0))
        out_list=[]
        seq_len = x.size(1)#30
        l_list = []
        for t in range(seq_len):
            output_t = []
            for layer_idx in range(self.num_layers):
                if 0==t:
                    ht_1, ct_1 = hidden_state[layer_idx][0],hidden_state[layer_idx][1].cuda()
                    attention_h=hidden_state[-1][0].cuda()
                else:
                    ht_1, ct_1 = hct_1[layer_idx][0].cuda(),hct_1[layer_idx][1].cuda()
                if 0==layer_idx:
                    feature_map=self.conv(x[:, t, :, :, :]).cuda()
                    feature_map=feature_map.view(feature_map.size(0),feature_map.size(1),-1).cuda()
                    attention_map=self.Wha(attention_h).cuda()
                    attention_map=torch.unsqueeze(self.softmax(attention_map),1).cuda()
                    
                    attention_feature=attention_map.cuda()*feature_map.cuda()
                    attention_feature=torch.sum(attention_feature,2).cuda()
                    
                    ht, ct = self.cell_list[layer_idx](attention_feature.cuda(),ht_1.cuda(), ct_1.cuda())
                    output_t.append([ht.cuda(),ct.cuda()])
                else:
                    ht, ct = self.cell_list[layer_idx](output_t[layer_idx-1][0].cuda(), ht_1.cuda(), ct_1.cuda())
                    output_t.append([ht.cuda(),ct.cuda()])
            attention_h=output_t[-1][0].cuda()
            hct_1=output_t
            
            aaa = self.fc(output_t[-1][0]).cuda()
            aaa = torch.unsqueeze(aaa,0)
            bn = nn.BatchNorm1d(x.size(0))
            bn = bn.cuda()
            bbb = bn(aaa)
            bbb = torch.squeeze(bbb,0)
            out_list.append(self.soft_out(self.tanh(bbb)))  
            l_list.append(attention_map.cuda())

        return torch.stack(out_list,1),torch.stack(l_list,1).squeeze(2)


    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):            
            tensor1 = torch.empty(batch_size, self.hidden_num[i])
            tensor2 = torch.empty(batch_size, self.hidden_num[i])
            ts1 = nn.init.orthogonal_(tensor1)
            ts2 = nn.init.orthogonal_(tensor2)
            
            init_states.append([ts1,ts2])
        return init_states


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param 



def train_batch():
    model = ALSTM(512,[49]*1,1,5)    
        
    model = model.cuda()
    for para in model.conv.parameters():
        para.requires_grad = False
    learning_rate = 1e-3  

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    
    accumulation_steps = int(14448/4)
    
    for t in range(10): # 
        
        train_loss=[]
        train_acc = 0
        #model.zero_grad()   
        train_mean = 0
        loader = ite.dataiter()  
        t_start = time.time()                               # Reset gradients tensors
        for j in range(accumulation_steps):
            print ("epoch: ",t)
            x_train,y_train = loader.next()
            x_train = Variable(x_train,requires_grad=True )
            y_train = Variable(y_train,requires_grad=True )
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            optimizer.zero_grad()
            
            y_pred,l = model(x_train)                    # Forward pass            
            my_loss,my_acc = my_cross_loss(l,y_train,y_pred,1)
            my_loss = my_loss / accumulation_steps                # Normalize our loss (if averaged)            
            with open("./loss.txt",'a+') as f:
                f.writelines(str(my_loss.item()))
                f.writelines('\n')
            
            my_loss.backward()        
            optimizer.step()                            # Now we can do an optimizer step           
            train_acc = train_acc + my_acc
            train_mean = train_mean + my_loss.item()
            print (j,' loss,acc are ',my_loss.item(),my_acc)
        print ('mean_loss',train_mean)
        t_end = time.time()
        delta = t_end-t_start
        print ("training time ",delta)
        with open("./train_loss.txt",'a+') as f:

                f.writelines(str(train_mean))
                f.writelines('\n')
           
    
    
    
    
        
        train_acc =  train_acc / accumulation_steps
        
        print ('training accuracy is ',train_acc)    
        with open("./train_acc.txt",'a+') as f:
            f.writelines(str(train_acc))
            f.writelines('\n')
        
        
        torch.save(model, './model_cfair10_2.pth')    # save model
    
    





 
if __name__ == '__main__':    
    train_batch()
    #train_seg()
    model = ALSTM(512,[49]*1,1,5)
    print (model)
    
