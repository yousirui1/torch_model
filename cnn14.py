import torch
import os
os.sys.path.append('/home/ysr/project/ai/open_source/audioset_tagging_cnn/pytorch')
from models import Cnn14_16k  # 来自 PANNs 代码库
from torchinfo import summary
import torch.nn as nn

model = Cnn14_16k(
    sample_rate=16000,
    window_size=512,      # ~32ms
    hop_size=160,         # ~10ms
    mel_bins=64,
    fmin=50,
    fmax=8000,            # 覆盖人声+敲击高频
    classes_num=527
)
weight_path = '/home/ysr/project/models/torch/panns/Cnn14_16k_mAP=0.438.pth'
checkpoint = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(checkpoint['model'])

model.fc1 = nn.Linear(2048, 512, bias=True)
model.fc_audioset = nn.Linear(512, 527, bias=True)
model.init_weight()

print(summary(model, input_size=(1, 16000)))
#model.eval()

if torch.cuda.is_available():
    device = torch.device('cuda')  # 选择可用的 GPU 设备
    gpu_name = torch.cuda.get_device_name(device)
    print("GPU is:", gpu_name)
else:
    device = torch.device('cpu')
    print("GPU is Not found")


def export_onnx(model, onnx_weight_path):
    dummy_input = torch.randn(1, 16000)
    torch.onnx.export(
        model.to(torch.device('cpu')),
        dummy_input,
        onnx_weight_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['waveform'],
        output_names=['clipwise_output', 'embedding'],
        dynamic_axes={
            'waveform': {0: 'batch', 1: 'time'},
            'embedding': {0: 'batch'}  # PANNs 的 embedding 是 (B, 2048)
        }
    )
