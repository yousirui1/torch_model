import os
import logging
import torch
from train_utils.set_all_random_seed import set_all_random_seed

from .fsmn_kws.model import FsmnKWS, FsmnKWSConvert
from .fsmn_kws_mt.model import FsmnKWSMT, FsmnKWSMTConvert
from .fsmn_kws_streaming.model import FsmnKWSStreaming
from .farfield.model import FSMNSeleNetV2Decorator
from frontends import frontend_classes
from tokenizer import tokenizer_classes
from train_utils.load_pretrained_model import load_pretrained_model
import utils.export_utils 

model_classes = dict (
    FsmnKWS = FsmnKWS,
    FsmnKWSConvert = FsmnKWSConvert,
    FsmnKWSStreaming = FsmnKWSStreaming,

    FsmnKWSMT = FsmnKWSMT,
    FsmnKWSMTConvert = FsmnKWSMTConvert,

    DFsmnV2 = FSMNSeleNetV2Decorator,
)

class Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model(**kwargs)

    def build_model(self, **kwargs):
        set_all_random_seed(kwargs.get("seed", 0))
        torch.set_num_threads(kwargs.get("ncpu", 4))

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"

        kwargs["device"] = device

        frontend_class = frontend_classes.get(kwargs.get("frontend", None)) 
        if frontend_class is not None:
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = (
                frontend.output_size() if hasattr(frontend, "output_size") else None
            )
            kwargs["frontend"] = frontend

        tokenizer_class = tokenizer_classes.get(kwargs.get("tokenizer", None))
        if tokenizer_class is not None:
            print(kwargs.get("tokenizer_conf", {}))
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["tokenizer"] = tokenizer
            
        model_class = model_classes.get(kwargs.get("model", None))
        model = model_class(**kwargs)

        init_param = kwargs.get("init_param", None)
        if init_param is not None:
            if os.path.exists(init_param):
                logging.info(f"Loading pretrained params from {init_param}")
                load_pretrained_model(
                    model=model,
                    path=init_param,
                    ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                    oss_bucket=kwargs.get("oss_bucket", None),
                    scope_map=kwargs.get("scope_map", []),
                    excludes=kwargs.get("excludes", None),
                )

        model.to(device)
        #kwargs["vocab_size"] = 0
        kwargs["model"] = model
        return model, kwargs

    def export(self, input=None, **cfg):
        device = cfg.get("device", "cpu")

        model,_ = self.model
        model = model.to(device)

        # deep_update(kwargs, cfg)
        kwargs = self.kwargs
        del kwargs["model"]
        kwargs["device"] = device

        model.eval()

        data_list = None # to do 
        #key_list, data_list = prepare_data_iterator()

        with torch.no_grad():
            print('export_utils export ', model)
            export_dir = utils.export_utils.export(model=model, data_in=data_list, **kwargs)

        return export_dir

