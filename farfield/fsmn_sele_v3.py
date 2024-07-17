import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsmn import AffineTransform, Fsmn, LinearTransform, RectifiedLinear

class DFSMNUnit(nn.Module):
    """ one multi-channel deep fsmn unit
    Args:
        dimin:                  input dimension
        dimexpand:              feature expansion dimension
        dimout:                 output dimension
        lorder:                 left ofder
        rorder:                 right order
    """

    def __init__(self,
                 dimin=64,
                 dimexpand=128,
                 dimout=64,
                 lorder=10,
                 rorder=1):
        super(DFSMNUnit, self).__init__()

        self.expand = AffineTransform(dimin, dimexpand)
        self.shrink = LinearTransform(dimexpand, dimout)
        self.fsmn = Fsmn(dimout, dimout, lorder, rorder, 1, 1)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        """
        Args:
            input: [batch, time, feature]
        """
        out1 = F.relu(self.expand(input))
        out2 = self.shrink(out1)
        out3 = self.fsmn(out2)

        # add skip connection for matched data
        if input.shape[-1] == out3.shape[-1]:
            out3 = input + out3
        if self.debug:
            self.dataout = out3
        return out3


class FSMNSeleNetV3(nn.Module):
    """ Deep FSMN model with channel selection performs multi-channel kws.
    Zhang, Shiliang, et al. "Deep-FSMN for large vocabulary continuous speech
    recognition." 2018 IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP). IEEE, 2018.

    Args:
        input_dim:              input dimension
        linear_dim:             fsmn input dimension
        proj_dim:               fsmn projection dimension
        lorder:                 fsmn left order
        rorder:                 fsmn right order
        num_syn:                output dimension
        fsmn_layers:            no. of fsmn units
    """

    def __init__(self,
                 input_dim=120,
                 linear_dim=128,
                 proj_dim=64,
                 lorder=10,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=5):
        super(FSMNSeleNetV3, self).__init__()

        self.mem = []
        # the first unit, mapping input dim to proj dim
        unit = DFSMNUnit(input_dim, linear_dim, proj_dim, lorder, rorder)
        self.mem.append(unit)
        self.add_module('mem_{:d}'.format(0), unit)

        # deep fsmn layers with skip connection
        for i in range(1, fsmn_layers):
            unit = DFSMNUnit(proj_dim, linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(i), unit)

        self.expand2 = AffineTransform(proj_dim, linear_dim)
        self.decision = AffineTransform(linear_dim, num_syn)

    def forward(self, input):
        # multi-channel temp space, [batch, time, channel, feature]
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.expand2.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.expand2.linear.out_features)

        for n in range(input.shape[2]):
            chin = input[:, :, n, :]

            for unit in self.mem:
                chout = unit(chin)
                chin = chout

            x[:, :, n, :] = F.relu(self.expand2(chout))

        # perform max pooling
        pool = nn.MaxPool2d((x.shape[2], 1), stride=(x.shape[2], 1))
        y = pool(x)

        # remove channel dimension
        y = torch.squeeze(y, -2)
        z = self.decision(y)

        return z

