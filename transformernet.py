import torch
from typing import Union, Dict, List, Tuple, Optional
from sanm.encoder import SANMEncoder
from paraformer.decoder import ParaformerSANMDecoder
from paraformer.cif_predictor import CifPredictorV2
from transformer.utils.nets_utils import make_pad_mask
from label_smoothing_loss import LabelSmoothingLoss
from transformer.utils.add_sos_eos import add_sos_eos
from transformer.utils.nets_utils import th_accuracy
from loss import mae_loss

class Transformer(torch.nn.Module):
    def __init__(
            self,
            input_size ,
            encoder: str,
            encoder_conf: Optional[Dict],
            decoder: str,
            decoder_conf: Optional[Dict],
	    predictor,
            predictor_conf: Optional[Dict],
            ctc: str,
            ctc_conf: Optional[Dict],
            vocab_size: int,
            ctc_weight: float = 0.5,
            ignore_id: int = -1,
            blank_id: int = 0,
            sos: int = 1,
            eos: int = 2,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            predictor_weight: float = 0.0,
            predictor_bias: int = 0,
            sampling_ratio: float = 0.2,
    ):
        super().__init__()

        if encoder == 'transformer':
            self.encoder = None
        elif encoder == 'conformer':
            self.encoder = None
        elif encoder == 'SANMEncoder':
            self.encoder = SANMEncoder(input_size=input_size, **encoder_conf)

        encoder_output_size = self.encoder.output_size()

        if decoder == 'transformer':
            self.decoder = None
        elif decoder == 'ParaformerSANMDecoder':
            self.decoder = ParaformerSANMDecoder(
                        vocab_size=vocab_size,             
                        encoder_output_size=encoder_output_size,
                        **decoder_conf
            )

        if predictor == 'CifPredictorV2':
            self.predictor = CifPredictorV2(**predictor_conf)

        if ctc == 'CTC':
            ctc = CTC(**ctc_conf)

        self.criterion_att = LabelSmoothingLoss(
                size = vocab_size,
                padding_idx = ignore_id,
                smoothing = lsm_weight,
                normalize_length = length_normalized_loss,
        )
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.ignore_id = ignore_id
        self.blank_id = blank_id
        self.sos = sos
        self.eos = eos
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
	
    def calc_predictor(self, encoder_out, encoder_out_lens):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id
        )
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def forward(self, x, x_length):
        encoder_out, encoder_out_lens,_ = self.encoder(x, x_length)

        # CTC branch
        if self.ctc_weight != 0.0:
            #loss_ctc, cer_ctc = self._calc_ctc_loss(
            #
            #)
            pass

        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )
        decoder_out, ys_pad_lens = self.decoder(encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length)

        return decoder_out, ys_pad_lens

    def calc_loss(self, 
            decoder_out: torch.Tensor,
            decoder_token_length: torch.Tensor,              # decoder token length 
            ys_pad: torch.Tensor,
            ys_pad_len: torch.Tensor
    ):
        #print(decoder_out)
        #print(decoder_token_length)
        #print(ys_pad)
        # ys_pad 对齐 ys_pad_len
        if self.predictor_bias == 1:
    #        _,ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad = add_sos_eos(ys_pad, decoder_token_length[decoder_token_length.argmax(0)], self.sos, self.eos, self.ignore_id)
    #        def add_sos_eos(ys_pad, max_pad_len, sos, eos, ignore_id):

        loss_pre = self.criterion_pre(decoder_token_length.type_as(ys_pad_len), ys_pad_len)

        loss_att = self.criterion_att(decoder_out, ys_pad)

	# 3. CTC-Att loss definition
        if self.ctc_weight == 0.0: 
            loss = loss_att + loss_pre * self.predictor_weight
        elif self.ctc_weight == 1.0: 
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight

        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        return loss, acc_att

    def calc_acc(self,
                decoder_out: torch.Tensor,
                ys_pad: torch.Tensor):
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        return loss, acc_att


