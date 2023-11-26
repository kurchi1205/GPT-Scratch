import torch
import json
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(onnx_path, quantized_onnx_path):
    torch_model = torch.load("models/gpt_trained_16000.pth")
    config = json.load(open("config.json"))
    stoi = config["stoi"]
    text =  ""
    encode = lambda s: [stoi[c] for c in s]
    encoded_text = torch.tensor(([encode(text)]), dtype=torch.long)
    torch_model = torch_model.to("cpu")
    encoded_text = encoded_text.to("cpu")
    onnx_model = torch.onnx.export(torch_model, encoded_text, input_names=["x"], output_names=["logits"],
                                   dynamic_axes={'x': {1: "length"}, "logits": {1: "length"}}, f=onnx_path)
    quantized_model = quantize_dynamic(onnx_path, quantized_onnx_path)

if __name__ == "__main__":
    export_to_onnx("torch_model.onnx", "model_quantized.onnx")