import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        output = self.linear(input)

        if self.debug:
            self.dataout = output

        return output


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        output = self.linear(input)

        if self.debug:
            self.dataout = output

        return output

class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input)


class Fsmn(nn.Module):
    """
    FSMN implementation.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 rorder=None,
                 lstride=None,
                 rstride=None):
        super(Fsmn, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim,
            self.dim, (lorder, 1),
            dilation=(lstride, 1),
            groups=self.dim,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim,
                self.dim, (rorder, 1),
                dilation=(rstride, 1),
                groups=self.dim,
                bias=False)
        else:
            self.conv_right = None

        self.debug = False
        self.dataout = None

    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = out1.squeeze(1)

        if self.debug:
            self.dataout = output

        return output


class FSMNNet(nn.Module):
    """
    FSMN net for keyword spotting
    """

    def __init__(self,
                 input_dim=200,
                 linear_dim=128,
                 proj_dim=128,
                 lorder=10,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=4):
        """
        Args:
            input_dim:              input dimension
            linear_dim:             fsmn input dimension
            proj_dim:               fsmn projection dimension
            lorder:                 fsmn left order
            rorder:                 fsmn right order
            num_syn:                output dimension
            fsmn_layers:            no. of sequential fsmn layers
        """
        super(FSMNNet, self).__init__()

        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.num_syn = num_syn
        self.fsmn_layers = fsmn_layers

        self.linear1 = AffineTransform(input_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

        self.fsmn = self._build_repeats(linear_dim, proj_dim, lorder, rorder,
                                        fsmn_layers)

        self.linear2 = AffineTransform(linear_dim, num_syn)


    @staticmethod
    def _build_repeats(linear_dim=136,
                       proj_dim=68,
                       lorder=3,
                       rorder=2,
                       fsmn_layers=5):
        repeats = [
            nn.Sequential(
                LinearTransform(linear_dim, proj_dim),
                Fsmn(proj_dim, proj_dim, lorder, rorder, 1, 1),
                AffineTransform(proj_dim, linear_dim),
                RectifiedLinear(linear_dim, linear_dim))
            for i in range(fsmn_layers)
        ]

        return nn.Sequential(*repeats)

    def forward(self, input):
        x1 = self.linear1(input)
        x2 = self.relu(x1)
        x3 = self.fsmn(x2)
        x4 = self.linear2(x3)
        return x4


class DFSMN(nn.Module):
    """
    One deep fsmn layer
    """

    def __init__(self,
                 dimproj=64,
                 dimlinear=128,
                 lorder=20,
                 rorder=1,
                 lstride=1,
                 rstride=1):
        """
        Args:
            dimproj:                projection dimension, input and output dimension of memory blocks
            dimlinear:              dimension of mapping layer
            lorder:                 left order
            rorder:                 right order
            lstride:                left stride
            rstride:                right stride
        """
        super(DFSMN, self).__init__()

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.expand = AffineTransform(dimproj, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)

        self.conv_left = nn.Conv2d(
            dimproj,
            dimproj, (lorder, 1),
            dilation=(lstride, 1),
            groups=dimproj,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                dimproj,
                dimproj, (rorder, 1),
                dilation=(rstride, 1),
                groups=dimproj,
                bias=False)
        else:
            self.conv_right = None

    def forward(self, input):
        f1 = F.relu(self.expand(input))
        p1 = self.shrink(f1)

        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = input + out1.squeeze(1)

        return output
