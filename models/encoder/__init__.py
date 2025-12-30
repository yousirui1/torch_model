from ..fsmn_kws.encoder import FSMNConvert
from ..fsmn_kws_mt.encoder import FSMNMT, FSMNMTConvert
#from ..fsmn_vad_streaming.encoder import FSMN, FSMNExport
from ..fsmn_kws_streaming.encoder import FSMN, FSMNExport
from ..farfield.fsmn_sele_v2 import FSMNSeleNetV2

encoder_classes = dict(
    FSMN = FSMN,
    FSMNExport = FSMNExport,

    FSMNConvert = FSMNConvert,

    FSMNMT = FSMNMT,
    FSMNMTConvert = FSMNMTConvert,

    FSMNSeleNetV2 = FSMNSeleNetV2,
)

