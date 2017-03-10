import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        if self.args.cuda: #Hack> Need to explicitly move weight, bias matrices to cuda
            for conv in self.convs1:
                conv.weight = torch.nn.Parameter(conv.weight.data.cuda())
                conv.bias = torch.nn.Parameter(conv.bias.data.cuda())
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    '''
    def conv_and_pool(self, x, conv):
        c = conv(x)
        x = F.relu(c).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    '''

    def forward(self, x):
        #print(x.size())
        x = self.embed(x) # (N,W,D)

        if self.args.static:
            x = torch.autograd.Variable(x)

        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.convs1[0]) #(N,Co)
        x2 = self.conv_and_pool(x,self.convs1[1]) #(N,Co)
        x3 = self.conv_and_pool(x,self.convs1[2]) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit