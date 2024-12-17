import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling, or last-step pooling. In case of bidirectional LSTMs
    last-step-bi pooling should be used instead of last-step pooling.
    '''
    def __init__(self,
                 d_input,
                 output_size=1,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 num_layer=1
                 ):
        super().__init__()

        if pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input, output_size)
            else:
                self.model = PoolAttFF(d_input, output_size, h=att_h, dropout=att_dropout)
        elif pool=='last_step_bi':
            self.model = PoolLastStepBi(d_input, output_size)
        elif pool=='last_step':
            self.model = PoolLastStep(d_input, output_size)
        elif pool=='max':
            self.model = PoolMax(d_input, output_size)
        elif pool=='avg':
            self.model = PoolAvg(d_input, output_size)
        else:
            raise NotImplementedError('Pool option not available')

        if num_layer > 1:
            self.model = self._get_clones(self.model, num_layer)
        self.d_input = d_input

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def get_model(self):
        return self.model

    def forward(self, x, n_wins):
        return self.model(x, n_wins)


class PoolLastStepBi(nn.Module):
    '''
    PoolLastStepBi: last step pooling for the case of bidirectional LSTM
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x.view(x.shape[0], n_wins.max(), 2, x.shape[-1]//2)
        x = torch.cat(
            (x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1, 0, :],
            x[:,0,1,:]),
            dim=1
            )
        x = self.linear(x)
        return x

class PoolLastStep(nn.Module):
    '''
    PoolLastStep: last step pooling can be applied to any one-directional 
    sequence.
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1]
        x = self.linear(x)
        return x

class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
        att = self.linear1(x)
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        x = self.linear2(x)

        return x

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''
    def __init__(self, d_input, output_size, h, dropout=0.1):
        super().__init__()
        self.d_input = d_input

        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)

        self.linear3 = nn.Linear(d_input, output_size)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_wins):
        x = x.view(x.shape[0], -1, self.d_input)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        #print(mask.shape)
        #mask = mask.squeeze(0)  # to do 
        #print(n_wins.shape)
        #print('-----------', mask.shape)
        #print(mask)
        #print(att.shape)
        #print(mask.unsqueeze(1).shape)
        #att[~mask.unsqueeze(1)] = float("-Inf")
        att[~mask] = float("-Inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        x = self.linear3(x)
        return x

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))

        x = self.linear(x)

        return x

class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]

        x = self.linear(x)

        return x

