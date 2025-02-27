import torch
from transformer.utils.nets_utils import pad_list

def add_sos_eos_base(ys_pad, sos, eos, ignore_id):
    '''
    Add <sos> and <eos> labels.
    param:
        ys_pad: torch.Tensor batch of padded target sequences (B, Lmax)
        sos: int index of <sos>
        eos: int index of <eos>
        ignore_id: int index of padding
    return:
        padding: tensor (B, Lmax)
    '''
    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_sos_eos(ys_pad, max_pad_len, sos, eos, ignore_id):
    ''' 
    Add <sos> and <eos> labels.
    param:
        ys_pad: torch.Tensor batch of padded target sequences (B, Lmax)
        sos: int index of <sos>
        eos: int index of <eos>
        ignore_id: int index of padding
    return:
        padding: tensor (B, Lmax)
    '''

    #ys_out = torch.zeron([len(ys_pad), max_pad_len])
    sos = torch.tensor([sos])
    eos = torch.tensor([eos])

    if max_pad_len == 0:
        max_pad_len = max(y.size(0) for y in ys_pad)
    
    y_out = [torch.cat([sos, torch.as_tensor(y), eos], dim=0) for y in ys_pad]

    n_batch = len(y_out)
    pad = y_out[0].new(n_batch, int(max_pad_len.item()), *y_out[0].size()[1:]).fill_(ignore_id)

    for i in range(n_batch):
        # to do
        if y_out[i].size(0) > max_pad_len:
            pad[i, : max_pad_len] = y_out[i][0:max_pad_len]
        else:
            pad[i, : y_out[i].size(0)] = y_out[i]

    return pad


