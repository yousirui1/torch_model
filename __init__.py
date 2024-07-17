import os
import torch
import torch.nn  as nn
from torchvision.models import mobilenet_v2

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
os.sys.path.append(parent_dir)

from .efficient import efficient
from .nearfield.fsmn import FSMN
from .nearfield.cmvn import GlobalCMVN, load_cmvn
from .farfield.fsmn_sele_v2 import FSMNSeleNetV2
from .farfield.fsmn_sele_v3 import FSMNSeleNetV3
from .yamnet import YAMNet
from .autoencoder import AutoEncoder
from .model import Model
from .attention import MHeadAttention, Attention, MeanPooling

def get_model(config, pretrain=False):
    global_cmvn = None
    preprocessing = None
    backbone = None
    classifier = None
    attention = None
    activation = None

    att_dim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]

    if config.preprocessing:
        preprocessing = None

    if config.cmvn_file != '':
        mean, istd = load_cmvn(config.cmvn_file)
        global_cmvn = GlobalCMVN(
                        torch.from_numpy(mean).float(),
                        torch.from_numpy(istd).float(),
                        )
    
    if config.backbone == 'fsmn':
        backbone = FSMN(config.idim, 140, 4, 250, config.hdim, 10, 2, 1, 1, 140, config.odim)
        #pretrain_model_path = parent_dir + '/pretrained_models/fsmn.pth'
    elif config.backbone == 'yamnet':
        backbone = YAMNet()
        pretrain_model_path = parent_dir + '/models/pretrained_models/yamnet.pth'
    elif config.backbone == 'efficient':
        backbone = efficient(2, False)
        pretrain_model_path = parent_dir + '/models/pretrained_models/fsd_efficient.pth'
    elif config.backbone == 'mobilenet_v2':
        backbone = mobilenet_v2(weights=None)
        backbone.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif config.backbone == 'autoencoder':
        backbone = AutoEncoder(config.idim, config.hdim)

    if pretrain and pretrain_model_path is not None:
        print('load pretrain_model_path ', pretrain_model_path)
        backbone.load_state_dict(torch.load(pretrain_model_path, map_location='cpu')) 

    if config.classifier == 'linear':
        classifier = nn.Linear(backbone.classifier.in_features, config.odim, bias=True)

    if config.classifier == 'avgpool':
        classifier = nn.AvgPool2d((4, 1))

    if config.attention == 1:
        attention = MHeadAttention(
                    att_dim[2], config.odim,
                    att_activation='relu',
                    cla_activation='relu')
    elif config.attention > 1:
        attention = Attention(
                    att_dim[2], config.odim,
                    att_activation='relu',
                    cla_activation='relu')

    if config.activation == 'softmax':
        activation = torch.torch.nn.functional.softmax

    elif config.activation == 'sigmoid':
        activation = torch.sigmoid

    model = Model(config.idim, config.odim, config.hdim,
           global_cmvn, preprocessing, backbone, classifier, attention, activation, config.is_cache)

    return model

