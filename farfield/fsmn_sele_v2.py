import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsmn import AffineTransform, Fsmn, LinearTransform, RectifiedLinear

class FSMNUnit(nn.Module):
    """ A multi-channel fsmn unit

    """

    def __init__(self, dimlinear=128, dimproj=64, lorder=20, rorder=1):
        """
        Args:
            dimlinear:              input / output dimension
            dimproj:                fsmn input / output dimension
            lorder:                 left order
            rorder:                 right order
        """
        super(FSMNUnit, self).__init__()

        self.shrink = LinearTransform(dimlinear, dimproj)
        self.fsmn = Fsmn(dimproj, dimproj, lorder, rorder, 1, 1)
        self.expand = AffineTransform(dimproj, dimlinear)

        self.debug = False
        self.dataout = None

    ''' 
    batch, time, channel, feature
    '''

    def forward(self, input):
        if torch.cuda.is_available():
            out = torch.zeros(input.shape).cuda()
        else:
            out = torch.zeros(input.shape)

        for n in range(input.shape[2]):
            out1 = self.shrink(input[:, :, n, :]) 
            out2 = self.fsmn(out1)
            out[:, :, n, :] = F.relu(self.expand(out2))

        if self.debug:
            self.dataout = out 

        return out 


class FSMNSeleNetV2(nn.Module):
    """ FSMN model with channel selection.
    """

    def __init__(self,
                 input_dim=120,
                 linear_dim=128,
                 proj_dim=64,
                 lorder=20,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=5,
                 sele_layer=0):
        """
        Args:
            input_dim:              input dimension
            linear_dim:             fsmn input dimension
            proj_dim:               fsmn projection dimension
            lorder:                 fsmn left order
            rorder:                 fsmn right order
            num_syn:                output dimension
            fsmn_layers:            no. of fsmn units
            sele_layer:             channel selection layer index
        """
        super(FSMNSeleNetV2, self).__init__()

        self.sele_layer = sele_layer

        self.featmap = AffineTransform(input_dim, linear_dim)

        self.mem = []
        for i in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(i), unit)

        self.decision = AffineTransform(linear_dim, num_syn)

    def forward(self, input):
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.featmap.linear.out_features)

        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))

        for i, unit in enumerate(self.mem):
            y = unit(x)

            # perform channel selection
            if i == self.sele_layer:
                pool = nn.MaxPool2d((y.shape[2], 1), stride=(y.shape[2], 1))
                y = pool(y)

            x = y

        # remove channel dimension
        y = torch.squeeze(y, -2)
        z = self.decision(y)

        return z

