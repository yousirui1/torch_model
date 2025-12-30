import types
import torch

def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"

    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)
    return model


def export_forward(self, feats: torch.Tensor, *args, **kwargs):
    scores = self.encoder(feats, *args)
    return scores

def export_dummy_inputs(self, data_in=None, frame=30):
    if data_in is None:
        speech = torch.randn(1, frame, 1, self.encoder_conf.get("input_dim"))
    else:
        speech = None # Undo
    return (speech)

def export_input_names(self):
    return ["speech"]


def export_output_names(self):
    return ["logits"]

def export_dynamic_axes(self):
    return {
        "speech": {1: "feats_length"},
    }

def export_name(self):
    return "model.onnx"
