import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader
from torchvision.models import efficientnet_b0
import cv2
import os
import numpy as np
import albumentations as A

b = 32
n = 6
k = 1
h = w = 105
d = 1280
heads = 4
dropout = 0.1
layers = 3

class ImgDataset(Dataset):
    def __init__(self, root_dir, n, k, mul, transform=None):
        self.root_dir = root_dir
        self.dirs = self.extract_from_dir(root_dir)
        self.transform = transform
        self.n = n
        self.k = k
        self.mul = mul
        self.T = len(self.dirs)

    @staticmethod
    def extract_from_dir(root_dir):
        dirs = []
        for T in os.listdir(root_dir):
            t = []
            for C in os.listdir(os.path.join(root_dir, T)):
                c = []
                for example in os.listdir(os.path.join(root_dir, T, C)):
                    c.append(os.path.join(root_dir, T, C, example))
                t.append(c)
            dirs.append(t)
        return dirs

    @staticmethod
    def transform_char(imgs):
        # apply same transform to all examples in the same char
        # imgs has shape k,h,w
        if np.random.rand() < 0.5:
            imgs = imgs[:,::-1] # v-flip
        if np.random.rand() < 0.5:
            imgs = imgs[:,:,::-1] # h-flip
        return imgs
    
    def sample(self,t):
        # t is self.dirs[idx]
        out = []
        for n_t in np.random.choice(len(t),self.n,replace=False):
            temp = []
            for n_exp in np.random.choice(20,self.k+1,replace=False):
                file_text = t[n_t][n_exp]
                img = self.load_transform(file_text)
                temp.append(img)
            temp = self.transform_char(np.stack(temp,0))
            out.append(temp)
        return out
    
    def load_transform(self,path):
        # apply per img transform here
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img /= 255.0
        if self.transform:
            img = self.transform(image=img)["image"]
        return img
    
    def __len__(self):
        return self.T * self.mul # to ensure larger batch_size than len(self.dirs)

    def __getitem__(self, idx):
        imgs = self.sample(self.dirs[idx%self.T])
        imgs = np.stack(imgs,0)
        imgs = imgs.reshape(self.n*(self.k+1),105,105)
        return imgs

class feedfwd(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(d, d),nn.ELU(),nn.Linear(d, d))
        self.LayerNorm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class Atten_fwd(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.atten = nn.MultiheadAttention(d,heads,dropout,batch_first=True)
        self.fwd = feedfwd(d,dropout)
    
    def forward(self, out):
        temp = self.atten(out,out,out,need_weights=False)[0]
        out = self.fwd(temp,out)
        return out
    
class AttenMeta_linearHead(nn.Module):
    def __init__(self, d, dropout,layers,usePreTrain=False):
        super().__init__()
        cnn = efficientnet_b0(weights='IMAGENET1K_V1' if usePreTrain else None)
        cnn.classifier = nn.Identity()
        self.cnn = cnn
        self.atten = nn.Sequential(*[Atten_fwd(d,dropout) for _ in range(layers)])
        self.linear = nn.Linear(d,n)
        self.loss_fn = nn.CrossEntropyLoss()

        self.input_target = Parameter(torch.randn(n,3)/n)
        y = torch.arange(n).reshape(1,n).broadcast_to(b,n).reshape(b*n)
        self.register_buffer('y',y)
        #self.register_parameter('input_target',input_target)

    def forward(self, x, IsTrain):
        # x with shape (b,n*(k+1),h,w)
        input_target = self.input_target.reshape(1,n,1,3,1,1).broadcast_to(b,n,k+1,3,h,w).reshape(b*n*(k+1),3,h,w)
        x = x.reshape(b*n*(k+1),1,h,w).broadcast_to(b*n*(k+1),3,h,w) + input_target# broadcast_to to 3 as model needs (R,G,B)
        out = self.cnn(x)
        out = out.reshape(b,n*(k+1),-1)
        out = self.atten(out).reshape(b,n,k+1,d)[:,:,-1,:].reshape(b*n,d)
        yhat = self.linear(out)

        if IsTrain:
            loss = self.loss_fn(yhat,self.y)
            return loss
        else:
            return yhat
        
class AttenMeta_attenHead(nn.Module):
    def __init__(self, d, dropout,layers,usePreTrain=False):
        super().__init__()
        cnn = efficientnet_b0(weights='IMAGENET1K_V1' if usePreTrain else None)
        cnn.classifier = nn.Identity()
        self.cnn = cnn
        self.atten = nn.Sequential(*[Atten_fwd(d,dropout) for _ in range(layers)])
        self.linear = nn.Linear(d,d)
        self.loss_fn = nn.CrossEntropyLoss()

        value = torch.eye(n)
        value = value.reshape(n,1,n).broadcast_to(n,k,n).reshape(n*k,n)
        y = torch.arange(n).reshape(1,n).broadcast_to(b,n).reshape(b*n)
        self.register_buffer('value',value)
        self.register_buffer('y',y)

    def forward(self, x, IsTrain):
        # x with shape (b,n*(k+1),h,w)
        x = x.reshape(b*n*(k+1),1,h,w).broadcast_to(b*n*(k+1),3,h,w) # broadcast_to to 3 as model needs (R,G,B)
        out = self.cnn(x)
        out = out.reshape(b,n*(k+1),-1)
        out = self.atten(out)
        out = self.linear(out)

        key, query = out.reshape(b,n,k+1,d).split([k,1], dim=2)
        key, query = key.reshape(b,n*k,d),query.reshape(b,n,d)
        weight = torch.einsum('bnd,bmd->bnm',query,key) # no softmax as CrossEntropyLoss expects logits
        yhat = torch.einsum('bnm,mq->bnq',weight,self.value)/k/n
        if IsTrain:
            yhat  = yhat.reshape(b*n,n)
            loss = self.loss_fn(yhat,self.y)
            return loss
        else:
            return yhat
