import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def ctc_loss(logits: torch.Tensor,
             target: torch.Tensor,
             logits_lengths: torch.Tensor,
             target_lengths: torch.Tensor,
             need_acc: bool = False):
    """ CTC Loss
    Args:
        logits: (B, D), D is the number of keywords plus 1 (non-keyword)
        target: (B)
        logits_lengths: (B)
        target_lengths: (B)
    Returns:
        (float): loss of current batch
    """

    acc = 0.0 
    if need_acc:
        acc = acc_utterance(logits, target, logits_lengths, target_lengths)

    # logits: (B, L, D) -> (L, B, D)
    logits = logits.transpose(0, 1)
    logits = logits.log_softmax(2)
    loss = F.ctc_loss(
        logits, target, logits_lengths, target_lengths, reduction='sum')
    # loss = loss / logits.size(1)
    return loss, acc 


def calc_alpha(sample_sizes, use_softmax=False):
    data_sizes = 1.0 / torch.tensor(sample_sizes)
    if use_softmax:
        data_sizes = F.softmax(data_sizes)

    else:
        data_sizes = data_sizes / data_sizes.sum()

    return data_sizes

class FocalLoss(nn.Module):
    def __init__(self, sample_sizes, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        alpha = calc_alpha(sample_sizes)
        class_num = len(alpha)
        if alpha is None:  # Now is impossible
            self.alpha = Variable(torch.ones(class_num, 1) / class_num)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.) 

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
