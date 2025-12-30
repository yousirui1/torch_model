import os
import torch

def export_onnx(
    model,
    data_in=None,
    quantize: bool = False,
    opset_version: int = 14,
    export_dir: str = None,
    **kwargs,
):

    device = kwargs.get("device", "cpu")
    dummy_input = model.export_dummy_inputs()

    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.to(device)
    else:
        dummy_input = tuple([input.to(device) for input in dummy_input])

    verbose = kwargs.get("verbose", False)

    if isinstance(model.export_name, str):
        export_name = model.export_name + ".onnx"
    else:
        export_name = model.export_name()
    model_path = os.path.join(export_dir, export_name)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=verbose,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
    )
    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            import onnx
        except:
            raise RuntimeError(
                "You are quantizing the onnx model, please install onnxruntime first. via \n`pip install onnx`\n`pip install onnxruntime`."
            )

        quant_model_path = model_path.replace(".onnx", "_quant.onnx")
        onnx_model = onnx.load(model_path)
        nodes = [n.name for n in onnx_model.graph.node]
        nodes_to_exclude = [
            m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
        ]
        print("Quantizing model from {} to {}".format(model_path, quant_model_path))
        quantize_dynamic(
            model_input=model_path,
            model_output=quant_model_path,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            nodes_to_exclude=nodes_to_exclude,
        )


