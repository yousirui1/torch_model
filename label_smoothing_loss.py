import torch
from torch import nn
from transformer.utils.nets_utils import make_pad_mask

class LabelSmoothingLoss(nn.Module):
    '''
    Label-smoothing loss.
    size: int the number of class
    padding_idx: int ignored class id 
    smoothing: float smoothing rate (0.0 means the conventional CE)
    normalize_length: bool normalize loss by sequence length if True
    criterion: torch.nn.Module loss function to be smoothed
    '''
    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        ''' Construct an LabelSmoothingLoss object. '''
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        '''
        Compute loss between x and target.
        x: torch.Tensor prediction (batch, seqlen, class)
        target: torch.Tensor target signal masked with self.padding_id (batch, seqlen)

        return: scalar: float value
                rtype: torch.Tensor
        '''
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class SequenceBinaryCrossEntropy(nn.Module):
    def __init__(
        self,
        normalize_length=False,
        criterion=nn.BCEWithLogitsLoss(reduction='none')
    ):
        super().__init__()
        self.normalize_length = normalize_length
        self.criterion = criterion

    def forward(self, pred, label, lengths):
        pad_mask = make_pad_mask(lengths, maxlen=pred.shape[1]).to(pred.device)
        loss = self.criterion(pred, label)
        denom = (~pad_mask).sum() if self.normalize_length else pred.shape[0]
        return loss.masked_fill(pad_mask.unsqueeze(-1), 0).sum() / denom

class NllLoss(nn.Module):
    '''
    Nll loss.
    size: int the number of class
    padding_idx: int ignored class id
    normalize_length: bool normalize loss by sequence length if True
    criterion: torch.nn.Module loss function
    '''
    def __init__(
        self,
        size,
        padding_idx,
        normalize_length=False,
        criterion=nn.NLLLoss(reduction='none'),
    ):
        ''' Construct an NllLoss object. '''
        super(NllLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        '''
        Compute loss between x and target.

        param:
            x: torch.Tensor prediction (batch, seqlen, class)
            target: torch.Tensor target signal masked with self.padding_id(batch, seqlen)
        return:
            scalar: float value
            rtype: torch.Tensor
        '''
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            ignore = target == self.padding_idx # (B, )
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0) # avoid -1 index
        kl = self.criterion(x, target)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore, 0).sum() / denom


