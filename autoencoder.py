import torch
import torch.nn as nn
import torch.optim as optim

# 自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dims):
        super(AutoEncoder, self).__init__()

        # 编码器层
        encoder_layers = []
        input_size = input_shape[0] * input_shape[1]
        for i in range(len(hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_size, hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.input_shape = input_shape

        # 掩码层
        self.mask = nn.Linear(hidden_dims[-1], hidden_dims[-1])

        # 解码器层
        decoder_layers = []
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dims[0], input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成二维
        x = self.encoder(x)
        #mask = self.mask(x)  # 应用掩码层
        #masked_x = x * mask  # 掩码操作
        #x = self.decoder(masked_x)
        x = self.decoder(x)
        x = x.view(x.size(0), *self.input_shape)  # 还原成原始形状
        return x

if __name__ == '__main__':
    # 设置超参数
    input_shape = (128, 96)
    hidden_dims = [128, 128, 64, 64, 32]  # 多层隐含层
    batch_size = 8
    num_epochs = 100

    # 创建自编码器模型和损失函数
    model = AutoEncoder(input_shape, hidden_dims)
    test_data = torch.randn(1, *input_shape)

    onnx_path = "autoencoder.onnx"
    torch.onnx.export(model, test_data, onnx_path)
