import os.path as osp
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import hub
from attention import  MHeadAttention
#from ..params import YAMNetParams

__all__ = ['yamnet', 'yamnet_category_metadata']

ckpt_url = "https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth"

BATCHNORM_EPSILON = 1e-4

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF Slim
    """
    def __init__(self, *args, **kwargs):
        # remove padding argument to avoid conflict
        padding = kwargs.pop("padding", "SAME")
        # initialize nn.Conv2d
        super().__init__(*args, **kwargs)
        self.padding = padding
        assert self.padding == "SAME"
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input, self.weight, self.bias, self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups,
        )

    def tf_SAME_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]

        dilate = self.dilation
        dilate = dilate if isinstance(dilate, int) else dilate[dim]
        stride = self.stride
        stride = stride if isinstance(stride, int) else stride[dim]

        effective_kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        total_padding = max(
            0, (out_size - 1) * stride + effective_kernel_size - input_size
        )
        total_odd = int(total_padding % 2 != 0)
        return total_odd, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
        odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
        if odd_1 or odd_2:
            # NOTE: F.pad argument goes from last to first dim
            input = F.pad(input, [0, odd_2, 0, odd_1])

        return self.forward_func(
            input, padding=[ padding_1 // 2, padding_2 // 2 ]
        )


class CONV_BN_RELU(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(
            conv.out_channels, eps=BATCHNORM_EPSILON
        )  # NOTE: yamnet uses an eps of 1e-4. This causes a huge difference
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False
            )
        )

    def forward(self, x):
        return self.fused(x)


class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=input_dim, groups=input_dim,
                kernel_size=kernel, stride=stride,
                padding='SAME', bias=False,
            ),
        )
        self.pointwise_conv = CONV_BN_RELU(
            Conv2d_tf(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=1,
                padding='SAME', bias=False,
            ),
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self, activation='sigmoid'):
        super().__init__()
        net_configs = [
            # (layer_function, kernel, stride, num_filters)
            (Conv,          [3, 3], 2,   32),
            (SeparableConv, [3, 3], 1,   64),
            (SeparableConv, [3, 3], 2,  128),
            (SeparableConv, [3, 3], 1,  128),
            (SeparableConv, [3, 3], 2,  256),
            (SeparableConv, [3, 3], 1,  256),
            (SeparableConv, [3, 3], 2,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 1,  512),
            (SeparableConv, [3, 3], 2, 1024),
            (SeparableConv, [3, 3], 1, 1024)
        ]

        input_dim = 1
        self.layer_names = []
        for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(net_configs):
            name = 'layer{}'.format(i + 1)
            self.add_module(name, layer_mod(kernel, stride, input_dim, output_dim))
            input_dim = output_dim
            self.layer_names.append(name)
        
        self.classifier = nn.Linear(input_dim, 521, bias=True)
        self.activation = activation
        self.attention = None

    def forward(self, x, to_prob=False):
        x = torch.unsqueeze(x, dim=1)
        for name in self.layer_names:
            mod = getattr(self, name)
            x = mod(x)
        x = F.adaptive_avg_pool2d(x, 1)
        if self.attention is not None:
            x, norm_att = self.attention(x)
        else:
            x = x.reshape(x.shape[0], -1)
            x = self.classifier(x)

        if self.activation == 'softmax':
            out = torch.nn.functional.softmax(x, dim=1)
        elif self.activation == 'sigmoid':
            out = torch.sigmoid(x)
        return out


def yamnet_category_metadata():
    cat_meta_file = osp.join(
        osp.dirname(osp.realpath(__file__)), 'yamnet_category_meta.yml'
    )
    with open(cat_meta_file) as f:
        cat_meta = yaml.safe_load(f)
    return cat_meta


def yamnet(label_dim, activation, head_num=0, att_activation='relu', pretrained=True):
    model = YAMNet(activation)
    if pretrained:
        state_dict = hub.load_state_dict_from_url(ckpt_url, progress=True)
        model.load_state_dict(state_dict)
    model.classifier = nn.Linear(model.classifier.in_features, label_dim, bias=True)
    if head_num > 0:
        #to do
        middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        b = 0
        model.attention = MHeadAttention(
                                1024, label_dim,
                                att_activation = att_activation,
                                cla_activation = att_activation)
    return model


