import torch
from typing import Union, Dict, List, Tuple, Optional
# LabelSmoothingLoss
# mae_loss

class Paraformer(torch.nn.Module):
    def __init__(
        self,
        specaug = None,
        specaug_conf: Optional[Dict] = None,
        normalize = None,
        normalize_conf: Optional[Dict] = None,
        encoder = None,
        encoder_conf: Optional[Dict] = None,
        decoder = None,
        decoder_conf: Optional[Dict] = None,
        ctc = None,
        ctc_conf: Optional[Dict] = None,
        ctc_weight: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
    ):

        super().__init__()

        self.specaug = None
        self.normalize = None
        self.encoder = None
        self.decoder = None
        self.ctc = None
        #self.predictor = None

        if specaug is not None:
            self.specaug = specaug(**specaug_conf)

        if normalize is not None:
            self.normalize = normalize(**normalize_conf)

        self.encoder = encoder(input_size=input_size, **encoder_conf)
        encoder_output_size = self.encoder.output_size()

        if decoder is not None:
            self.decoder = decoder(
                vocab_size = vocab_size,
                encoder_output_size = encoder_output_size,
                **decoder_conf,
            )
        self.vocab_size = vocab_size

    def forward(
        self,
        speech: torch.Tensor,
        speech_length: torch.Tensor
    ):
        """
        Encoder + Decoder + Calc loss
        """

        if len(speech_length.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            # Normalizetion for feature: e.g Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)


        # Encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        #to do  
        # 未完成




