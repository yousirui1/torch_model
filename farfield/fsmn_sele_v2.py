import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsmn import AffineTransform, Fsmn, LinearTransform, RectifiedLinear
from .model_def import HEADER_BLOCK_SIZE, ActivationType, LayerType, f32ToI32

class FSMNUnit(nn.Module):
    """
    A multi-channel fsmn unit
    """
    def __init__(self, dimlinear=128, dimproj=64, lorder=20, rorder=1):
        """
        Args:
            dimlinear:      input / output dimension
            dimproj:        fsmn input / output dimension
            lorder:         left order
            rorder:         right order
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
    def forward(self, x):
        if torch.cuda.is_available():
            out = torch.zeros(x.shape).cuda()
        else:
            out = torch.zeros(x.shape)

        for n in range(x.shape[2]):
            out1 = self.shrink(x[:, :, n, :])
            out2 = self.fsmn(out1)
            out[:, :, n, :] = F.relu(self.expand(out2))

        if self.debug:
            self.dataout = out

        return out

    def print_model(self):
        self.shrink.print_model()
        self.fsmn.print_model()
        self.expand.print_model()

class FSMNSeleNetV2(nn.Module):
    """
    FSMN model with channel selection.
    """
    def __init__(self,
                input_dim=120,
                channel=1,
                linear_dim=128,
                proj_dim=64,
                lorder=20,
                rorder=1,
                num_syn=5,
                fsmn_layers=5,
                sele_layer=0):
        """
        Args:
            input_dim:          input dimension
            linear_dim:         fsmn input dimension
            proj_dim:           fsmn projection dimension
            lorder:             fsmn left order
            rorder:             fsmn right order
            num_syn:            output dimension
            fsmn_layers:        no. of fsmn units
            sele_layer:         channel selection layer index
        """
        super(FSMNSeleNetV2, self).__init__()
        self.sele_layer = sele_layer

        self.featmap = AffineTransform(input_dim, linear_dim)

        # y.shape[2] 
        self.pool = nn.MaxPool2d((channel, 1), stride=(channel, 1))
        self.mem = []

        for i in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(i), unit)

        self.decision = AffineTransform(linear_dim, num_syn)

    def forward(self, input, input_length=None):
        # [batch, time, channel, feature]
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features)

        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))


        for i, unit in enumerate(self.mem):
            y = unit(x)

            # perform channel selection
            if i == self.sele_layer:
                y = self.pool(y)

            x = y

        # remove channel dimension
        #print(y.shape)
        #y = y.reshape(1, -1)
        y = torch.squeeze(y, -2)
        print(y.shape)
        z = self.decision(y)

        return z

    def print_model(self):
        self.featmap.print_model()

        for unit in self.mem:
            unit.print_model()

        self.decision.print_model()

    def print_header(self):
        '''
        get FSMN params
        '''
        input_dim = self.featmap.linear.in_features
        linear_dim = self.featmap.linear.out_features
        proj_dim = self.mem[0].shrink.linear.out_features
        lorder = self.mem[0].fsmn.conv_left.kernel_size[0]
        rorder = 0
        if self.mem[0].fsmn.conv_right is not None:
            rorder = self.mem[0].fsmn.conv_right.kernel_size[0]

        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem)

        # no. of output channels, 0.0 mean the same as num_inputs
        num_outputs = 1.0

        # write total header
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # num_inputs
        header[0] = 0.0
        # num_outputs
        header[1] = num_outputs
        # idims
        header[2] = input_dim
        # odims
        header[3] = num_syn
        # num_layers
        header[4] = 3

        # write each layer's header
        header_index = 1

        header[HEADER_BLOCK_SIZE * header_index + 0] = float(
                LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * header_index + 1] = 0.0
        header[HEADER_BLOCK_SIZE * header_index + 2] = input_dim
        header[HEADER_BLOCK_SIZE * header_index + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * header_index + 4] = 1.0
        header[HEADER_BLOCK_SIZE * header_index + 5] = float(
                ActivationType.ACTIVATION_RELU.value)
        header_index += 1

        header[HEADER_BLOCK_SIZE * header_index + 0] = float(
                LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * header_index + 1] = 0.0
        header[HEADER_BLOCK_SIZE * header_index + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * header_index + 3] = proj_dim
        header[HEADER_BLOCK_SIZE * header_index + 4] = lorder
        header[HEADER_BLOCK_SIZE * header_index + 5] = rorder
        header[HEADER_BLOCK_SIZE * header_index + 6] = fsmn_layers
        if num_outputs == 1.0:
            header[HEADER_BLOCK_SIZE * header_index + 7] = float(self.sele_layer)
        else:
            header[HEADER_BLOCK_SIZE * header_index + 7] = -1.0
        header_index += 1

        header[HEADER_BLOCK_SIZE * header_index + 0] = float(
                LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * header_index + 1] = num_outputs
        header[HEADER_BLOCK_SIZE * header_index + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * header_index + 3] = num_syn
        header[HEADER_BLOCK_SIZE * header_index + 4] = 1.0
        header[HEADER_BLOCK_SIZE * header_index + 5] = float(
                ActivationType.ACTIVATION_SOFTMAX.value) 

        for h in header:
            print(f32ToI32(h))

