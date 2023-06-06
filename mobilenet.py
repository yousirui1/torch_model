import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchinfo import summary

class MobileNetV2(nn.Module):
    def __init__(self, input_dim, label_dim=527, pretrain=True):
        super(MobileNetV2, self).__init__()

        self.input_dim = input_dim
        self.model = mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        if len(self.input_dim) == 2:
            x = x.view(1, 1, self.input_dim[0], self.input_dim[1])
        elif len(self.input_dim) == 3:
            x = x.view(self.input_dim[0], 1, self.input_dim[1], self.input_dim[2])
        #x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out

if __name__ ==  '__main__':
    input_tdim = 1056
    #model = EffNetAttention(pretrain = False, reshape_dim = (1, 1, 128, 128), b = 0, head_num = 0)
    model = MobileNetV2(pretrain=False, input_dim = (1, input_tdim, 96))    
    test_input = torch.rand([1, input_tdim, 96])
    print(test_input.shape)
    summary(model, input_size=(1, input_tdim, 96))
    ouput = model(test_input)
    onnx_path = "efficient.onnx"
    torch.onnx.export(model, test_input, onnx_path)

