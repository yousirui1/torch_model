import torch
import torch.nn as nn
import torch.optim as optim

from mobilenet import MobileNetV2
from autoencoder import AutoEncoder

## Create the ensemble model
class EnsembleModel(nn.Module):
    def __init__(self, input_shape, hidden_dims, pretrain=True):
        super(EnsembleModel, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.mobilenet = MobileNetV2(input_dim = input_shape, label_dim = 128, pretrain=False)
        self.autoencoder = AutoEncoder(input_shape, hidden_dims)

    def forward(self, x): 
        x = self.autoencoder(x)
        #output = self.mobilenet(x)
        return x

        #autoencoder_output = self.autoencoder(x)
        #mobilenet_output = self.mobilenet(x)
        #ensemble_output = torch.mean(torch.stack([mobilenet_output, autoencoder_output]), dim=0)
        #return ensemble_output

if __name__ == '__main__':
    input_shape = (1568, 128)
    hidden_dims = [128, 128, 64, 64, 32]
    test_data = torch.randn(1, *input_shape)
    model = EnsembleModel(input_shape, hidden_dims)
    onnx_path = "ensemble.onnx"
    torch.onnx.export(model, test_data, onnx_path, input_names=['inputs'], 
            output_names=['outputs'])
