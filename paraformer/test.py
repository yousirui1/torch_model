import torch
from specaug import SpecAugLFR
from sanm.encoder import SANMEncoder
from decoder import ParaformerSANMDecoder
from cif_predictor import CifPredictorV2
from model import Paraformer

if __name__ == '__main__':
    input_tdim = 10
    input_ddim = 560 

    input_legnth = 10

    specaug = SpecAugLFR
    specaug_conf = {'apply_time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic', 'apply_freq_mask': True, 'freq_mask_width_range': [0, 30], 'lfr_rate': 6, 'num_freq_mask': 1, 'apply_time_mask': True, 'time_mask_width_range': [0, 12], 'num_time_mask': 1}

    normalize = None 
    normalize_conf = None 

    encoder = SANMEncoder
    encoder_conf = {'output_size': 512, 'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 50, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.1, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shift': 0, 'selfattention_layer_type': 'sanm'} 

    decoder = ParaformerSANMDecoder 
    decoder_conf = {'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 16, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1, 'att_layer_num': 16, 'kernel_size': 11, 'sanm_shift': 0} 
    ctc = None 
    ctc_conf =  {'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True} 
    predictor = CifPredictorV2
    predictor_conf = {'idim': 512, 'threshold': 1.0, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45} 
    ctc_weight = 0.0
    input_size = 560 

    vocab_size = 8404

    #model = Paraformer(specaug, specaug_conf, normalize, normalize_conf, encoder, encoder_conf, decoder, decoder_conf, ctc, ctc_conf, predictor, predictor_conf, ctc_weight, input_size, vocab_size)
    #print(model)

    encoder = encoder(input_size=input_size, **encoder_conf)

    print(encoder)
    input_data = torch.rand(1, 10, 560)
    #print(input_data)
    input_length = torch.tensor([10])
    print(input_data)
    encoder_out = encoder.forward(input_data, input_length)
    print(encoder_out)
    print(encoder_out[0].shape, encoder_out[1].shape)
    print(encoder_out[1])

