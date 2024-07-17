from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, 
                input: Tuple[torch.Tensor, torch.Tensor]):

        if isinstance(input, tuple):
            input, in_cache = input
        else:
            in_cache = torch.zeros(0, 0, 0, 0, dtype=torch.float)

        output = self.quant(input)
        output = self.linear(output)
        output = self.dequant(output)
        return (output, in_cache)

class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, 
                input: Tuple[torch.Tensor, torch.Tensor]):

        if isinstance(input, tuple):
            input, in_cache = input
        else:
            in_cache = torch.zeros(0, 0, 0, 0, dtype=torch.float)


        output = self.quant(input)
        output = self.linear(output)
        output = self.dequant(output)

        return (output, in_cache)

class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, 
            input: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(input, tuple):
            input, in_cache = input
        else :
            in_cache = torch.zeros(0, 0, 0, 0, dtype=torch.float)
        out = self.relu(input)            
        # out = self.dropout(out)
        return (out, in_cache)


class FSMNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lorder=None,
        rorder=None,
        lstride=1,
        rstride=1, 
    ):

        super(FSMNBlock, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim,
            self.dim, [lorder, 1],
            dilation=[lstride, 1],
            groups=self.dim,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim,
                self.dim, [rorder, 1],
                dilation=[rstride, 1],
                groups=self.dim,
                bias=False)
        else:
            self.conv_right = None

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, 
            input: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(input, tuple):
            input, in_cache = input
        else:
            in_cache = torch.zeros(0, 0, 0, 0, dtype=torch.float)

        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)

        if in_cache is None or len(in_cache) == 0 :
            x_pad = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride
                                  + self.rorder * self.rstride, 0])
        else:
            in_cache = in_cache.to(x_per.device)
            x_pad = torch.cat((in_cache, x_per), dim=2)

        in_cache = x_pad[:, :, -((self.lorder - 1) * self.lstride
                                 + self.rorder * self.rstride):, :]
        y_left = x_pad[:, :, :-self.rorder * self.rstride, :]
        #y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        y_left = self.quant(y_left)
        y_left = self.conv_left(y_left)
        y_left = self.dequant(y_left)
        #out = x_per + y_left
        out = x_pad[:, :, (self.lorder - 1) * self.lstride: -self.rorder *
                    self.rstride, :] + y_left

        if self.conv_right is not None:
            #y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = x_pad[:, :, -(
                x_per.size(2) + self.rorder * self.rstride):, :]

            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.quant(y_right)
            y_right = self.conv_right(y_right)
            y_right = self.dequant(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return (output, in_cache)
	

class FSMN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_affine_dim: int,
        fsmn_layers: int,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        output_affine_dim: int,
        output_dim: int,
    ):
        """
            Args:
                input_dim:              input dimension
                input_affine_dim:       input affine layer dimension
                fsmn_layers:            no. of fsmn units
                linear_dim:             fsmn input dimension
                proj_dim:               fsmn projection dimension
                lorder:                 fsmn left order
                rorder:                 fsmn right order
                lstride:                fsmn left stride
                rstride:                fsmn right stride
                output_affine_dim:      output affine layer dimension
                output_dim:             output dimension
        """

        super(FSMN, self).__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

        self.padding = (self.lorder - 1) * self.lstride \
            + self.rorder * self.rstride

        self.fsmn = self._build_repeats(fsmn_layers, linear_dim, proj_dim, lorder,
                                   rorder, lstride, rstride)

        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)

    def _build_repeats(
        self,
        fsmn_layers: int,
    	linear_dim: int,
    	proj_dim: int,
    	lorder: int,
    	rorder: int,
    	lstride=1,
    	rstride=1,
    ):
        repeats = [
            nn.Sequential(
                LinearTransform(linear_dim, proj_dim),
                FSMNBlock(proj_dim, proj_dim, lorder, rorder, 1, 1),
                AffineTransform(proj_dim, linear_dim),
                RectifiedLinear(linear_dim, linear_dim))
            for i in range(fsmn_layers)
        ]

        return nn.Sequential(*repeats)

    def forward(
        self,
        input: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            in_cache(torch.Tensor): (B, D, C), C is the accumulated cache size
        """
        if in_cache is None or len(in_cache) == 0 :
            in_cache = [torch.zeros(0, 0, 0, 0, dtype=torch.float)
                        for _ in range(len(self.fsmn))]
        else:
            print(in_cache)
            print('shape ',in_cache.shape)
            in_cache = [in_cache[:, :, :, i: i + 1] for i in range(in_cache.size(-1))]

        input = (input, in_cache)
        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        #x4 = self.fsmn(x3)
        x4, _ = x3
        for layer, module in enumerate(self.fsmn):
            x4, in_cache[layer] = module((x4, in_cache[layer]))

        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)
        # x7 = self.softmax(x6)
        x7, _ = x6

        return x7, torch.cat(in_cache, dim=-1)


def fsmn_test():
    fsmn = FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 2599)
    print(fsmn)

    num_params = sum(p.numel() for p in fsmn.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(128, 200, 400)  # batch-size * time * dim
    y, _ = fsmn(x)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))

    #print(fsmn.to_kaldi_net())
