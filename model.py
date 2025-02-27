import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
#from sanm.encoder import SANMEncoder
from yamnet import YAMNet
from efficient import Efficient
from pool import Pooling
from torchinfo import summary
from farfield.fsmn_sele_v2 import FSMNSeleNetV2
from farfield.fsmn_sele_v3 import FSMNSeleNetV3
from transformernet import Transformer
from nkf import KGNet
from farfield.fsmn import Fsmn

#from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))

class Model(nn.Module):
    def __init__(
        self, 
        idim: int,
        odim: int,
        embedding: nn.Module,
        classifier: nn.Module,
        activation: nn.Module,
        #is_use_cache: bool = False,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.embedding = embedding
        self.classifier = classifier
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        n_wins: torch.Tensor = None,  # data length mask compute
        #in_cache: torch.Tensor = torch.zeros(0, 0, 0, 0, dtype=torch.float)
    ):
        #print(n_wins)
        #print(torch.tensor([x.shape[1]]))
        #x_length = torch
        #x_length = []

        #for i in range(x.shape[0]):
        #    x_length.append(torch.tensor(x.shape[1]))
        #    if isinstance(x, tuple):
        #        x = x[0]
        #x_length = torch.tensor(x_length)

        if self.embedding is not None:
            x = self.embedding(x, n_wins)

        if self.classifier is not None:
            if isinstance(self.classifier, nn.ModuleList):
                out = [mod(x, n_wins) for mod in self.classifier]
                x = torch.cat(out, dim=1)
            elif isinstance(self.classifier, nn.Module):
                x = self.classifier(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

def get_model(batch_size, 
                idim, 
                odim, 
                embedding = None, 
                embedding_conf = None,
                classifier = None, 
                classifier_conf = None,
                activation = None,
                pretrained = False,
            ):

    pre_weight_path = None

    if embedding == "fsmn":
        embedding = Fsmn(**embedding_conf)
    elif embedding == 'fsmn_sele_v2':
        embedding = FSMNSeleNetV2(**embedding_conf)
    elif embedding == 'fsmn_sele_v3':
        embedding = FSMNSeleNetV3(**embedding_conf)
    elif embedding == 'kgnet':
        embedding = KGNet(**embedding_conf)
    elif embedding == "yamnet":
        embedding = YAMNet(**embedding_conf)
        pre_weight_path = current_dir + '/pretrained_models/yamnet.pth'
    elif embedding == "efficient":
        embedding = Efficient(**embedding_conf)
        pre_weight_path = current_dir + '/pretrained_models/fsd_efficient.pth'
    elif embedding == "autoencoder":
        embedding = Autoencoder(**embedding_conf)
    elif embedding == 'transformer':
        embedding = Transformer(input_size=idim, **embedding_conf)

    elif embedding == "SANMEncoder":
        embedding = SANMEncoder(input_size=idim, **embedding_conf)

    if pretrained and pre_weight_path is not None:
        print("embedding load weight ", pre_weight_path)
        embedding.load_state_dict(torch.load(pre_weight_path, map_location='cpu'))

    if classifier == 'linear':
        classifier = nn.Linear(embedding.classifier.in_features, odim, bias=True)
    elif classifier == 'pool':
        classifier = Pooling(**classifier_conf).get_model()
    elif classifier == 'attention':
        if classifier_conf.get("att_head", 1) > 1:
            classifier = MHeadAttention(**classifier_conf)
        else:
            classifier = Attention(**classifier_conf)
    else:
        classifier = None
    
    if activation == 'softmax':
        activation = torch.torch.nn.functional.softmax
    elif activation == 'sigmoid':
        activation = torch.sigmoid
    else:
        activation = None

    model = Model(idim, odim, embedding, classifier, activation)
    #summary(model, input_shape=[(1, 100, idim), (10)])
    return model

