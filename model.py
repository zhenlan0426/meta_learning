import torch
from torch import nn
from torchvision.models import efficientnet_b0

b = 2
n = 4
k = 3
h = w = 105
d = 1280
heads = 4
dropout = 0.1
layers = 2

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
    
class AttenMeta(nn.Module):
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
        

