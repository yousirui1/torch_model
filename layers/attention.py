import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    '''
    SelfAttention: The main SelfAttention module that can be used as a
    time-dependency model.                                            
    '''
    def __init__(self,
                input_size,
                 d_model=512,
                 nhead=8,
                 pool_size=3,
                 pos_enc=None,
                 num_layers=6,
                 sa_h=2048,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()

        encoder_layer = SelfAttentionLayer(d_model, nhead, pool_size, sa_h, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(input_size, d_model)

        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        if pos_enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = nn.Identity()

        self._reset_parameters()

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins=None):
        src = self.linear(src)
        output = src.transpose(1,0)
        output = self.norm1(output)
        output = self.pos_encoder(output)

        for mod in self.layers:
            output, n_wins = mod(output, n_wins=n_wins)
        return output.transpose(1,0), n_wins

class SelfAttentionLayer(nn.Module):
    '''
    SelfAttentionLayer: The SelfAttentionLayer that is used by the
    SelfAttention module.                                            
    '''
    def __init__(self, d_model, nhead, pool_size=1, sa_h=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

    def forward(self, src, n_wins=None):

        if n_wins is not None:
            mask = ~((torch.arange(src.shape[0])[None, :]).to(src.device) < n_wins[:, None].to(torch.long).to(src.device))
        else:
            mask = None

        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, n_wins

class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding: PositionalEncoding taken from the PyTorch Transformer
    tutorial. Can be applied to the SelfAttention module. However, it did not 
    improve the results in previous experiments.                          
    '''
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

