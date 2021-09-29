import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import numpy as np
import random
from transformer import Transformer

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
     
class ViTResNet(nn.Module):
    def __init__(self, block, num_blocks, batch_size):
        super(ViTResNet, self).__init__()
        self.in_planes = 32
        self.L = 16
        self.cT = 256
        self.mlp_dim = 512
        self.num_classes=5
        self.heads = 16
        self.depth = 16
        self.emb_dropout = 0.2
        self.transformer_dropout = 0.2
        self.input_channels = 10
        
        # Convolutions
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.apply(_weights_init)
        
        
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(batch_size,self.L, 128),requires_grad = True)
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(batch_size, 128, self.cT),requires_grad = True) 
        torch.nn.init.xavier_uniform_(self.token_wV)        
        
        self.pos_embedding = nn.Parameter(torch.empty(1, (self.L + 1), self.cT))
        torch.nn.init.normal_(self.pos_embedding, std = .02)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cT))
        self.dropout = nn.Dropout(self.emb_dropout)

        self.transformer = Transformer(self.cT, self.depth, self.heads, self.mlp_dim, self.transformer_dropout)

        self.to_cls_token = nn.Identity()
        
        self.nn_same = nn.Linear(self.cT, self.cT)
        torch.nn.init.xavier_uniform_(self.nn_same.weight)
        torch.nn.init.normal_(self.nn_same.bias, std = 1e-6)
        
        self.nn1 = nn.Linear(self.cT, self.num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        
    def forward(self, img, mask = None):
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x) 
        
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Tokenization 
        wa = rearrange(self.token_wA, 'b h w -> b w h') 
        A= torch.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)
        VV= torch.einsum('bij,bjk->bik', x, self.token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  

        # Class tokens and positional embeddings
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Attention
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])       
        x = self.nn_same(x)
        x = self.nn_same(x)
        x = self.nn1(x)
        
        return x
          
def data_loader(file_name, batch_size, device):
  file_names = open(file_name).read().split('\n')[:-1]
  total_samples = len(file_names) * 16
  random.shuffle(file_names)
  i = 0
  while 1:
    X = []
    Y = []
    k = batch_size
    
    while k > 0:
      cur_file = file_names[i]
      loaded = np.load(cur_file)
      X.append(loaded['X'])
      Y.append(loaded['y'])
      i += 1
      k -= 16
      
      if i >= len(file_names):
        i = 0
        random.shuffle(file_names)
        
    X = torch.from_numpy(np.concatenate(X)).float().to(device)
    y = torch.tensor(np.concatenate(Y)).long().to(device)

    yield total_samples, X, y

def train(model, optimizer, data_loader, loss_history, scheduler):
    model.train()
    epoch_loss = []
    correct_samples = 0
    i = 0
    
    for total_samples, data, target in data_loader:
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(target).sum()
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        epoch_loss.append(loss.item())

        if i % 10000 == 0:
          print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                '] Loss: ' + '{:6.4f}'.format(sum(epoch_loss)/len(epoch_loss)))
       
          for param_group in optimizer.param_groups:
            print('Learning rate: ', param_group['lr'])

        if i * len(data) >= total_samples:
          #one epoch completed
          break
        
        i += 1
        scheduler.step()
    
    print('\nTrain Loss: {:6.4f}'.format(sum(epoch_loss)/len(epoch_loss)) + 
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
            
def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = 0
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        i = 0
        for test_samples, data, target in data_loader:
            total_samples = test_samples
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
            
            if i * len(data) >= total_samples:
              #testing done
              break
            
            i += 1

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
    return ((100.0 * correct_samples / total_samples), avg_loss)
