import math
import struct
from enum import Enum

HEADER_BLOCK_SIZE = 10

class LayerType(Enum):
    LAYER_DENSE = 1
    LAYER_GRU = 2
    LAYER_ATTENTION = 3
    LAYER_FSMN = 4
    LAYER_SEQUENTIAL_FSMN = 5
    LAYER_FSMN_SELE = 6
    LAYER_GRU_ATTENTION = 7
    LAYER_DFSMN = 8
    LAYER_MAX_POOLING = 9

class ActivationType(Enum):
    ACTIVATION_NONE = 0
    ACTIVATION_RELU = 1
    ACTIVATION_TANH = 2
    ACTIVATION_SIGMOID = 3
    ACTIVATION_SOFTMAX = 4
    ACTIVATION_LOGSOFTMAX = 5


def i32Tof32(i):
    bs = struct.pack('i', i)

    return struct.unpack('f', bs)[0]

def f32ToI32(f):
    """
    print layer
    """
    bs = struct.pack('f', f)

    ba = bytearray()
    ba.append(bs[0])
    ba.append(bs[1])
    ba.append(bs[2])
    ba.append(bs[3])

    return struct.unpack('i', ba)[0]

def printNeonMatrix(w):
    """
    print matrix with neon padding
    """
    num_rows, num_cols = w.shape
    num_neon_cols = math.ceil(num_cols / 4)

    for i in range(num_rows):
        for j in range(num_cols):
            print(f32ToI32(w[i, j]))

        for j in range(num_neon_cols * 4 - num_cols):
            print(0)

def printNeonVector(b):
    """
    print vector with neon padding
    """
    size = b.shape[0]
    neon_size = math.ceil(size / 4)

    for i in range(size):
        print(f32ToI32(b[i]))

    for i in range(neon_size * 4 - size):
        print(0)

def printDense(layer):
    """
    print dense layer
    """
    state_dict = layer.state_dict()
    printNeonMatrix(state_dict['weight'])
    printNeonVector(state_dict['bias'])


def printGRU(layer):
    """
    save gru layer
    """
    state_dict = layer.state_dict()
    weight = [state_dict['weight_ih_l0'], state_dict['weight_hh_l0']]
    bias = [state_dict['bias_ih_l0'], state_dict['bias_hh_l0']]
    num_inputs, num_outputs = weight[0].shape
    num_inputs = num_inputs // 3

    # output input weights
    w_rx = weight[0][:num_inputs, :]
    w_zx = weight[0][num_inputs:num_inputs * 2, :]
    w_x = weight[0][num_inputs * 2:, :]
    printNeonMatrix(w_zx)
    printNeonMatrix(w_rx)
    printNeonMatrix(w_x)

    # output recurrent weights
    w_rh = weight[1][:num_inputs, :]
    w_zh = weight[1][num_inputs:num_inputs * 2, :]
    w_h = weight[1][num_inputs * 2:, :]
    printNeonMatrix(w_zh)
    printNeonMatrix(w_rh)
    printNeonMatrix(w_h)

    # output input bias
    b_rx = bias[0][:num_inputs]
    b_zx = bias[0][num_inputs:num_inputs * 2]
    b_x = bias[0][num_inputs * 2:]
    printNeonVector(b_zx)
    printNeonVector(b_rx)
    printNeonVector(b_x)

    # output input weights
    b_rh = bias[1][:num_inputs]
    b_zh = bias[1][num_inputs:num_inputs * 2]
    b_h = bias[1][num_inputs * 2:]
    printNeonVector(b_zh)
    printNeonVector(b_rh)
    printNeonVector(b_h)

