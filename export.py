import torch
import json

def export_to_onnx(onnx_path):
    torch_model = torch.load("models/gpt_trained_17000.pth")
    config = json.load(open("config.json"))
    stoi = config["stoi"]
    text =  "RHYME a a b c b c d e f d f e"
    encode = lambda s: [stoi[c] for c in s]
    encoded_text = torch.tensor(([encode(text)]), dtype=torch.long)
    torch_model = torch_model.to("cpu")
    encoded_text = encoded_text.to("cpu")
    onnx_model = torch.onnx.export(torch_model, encoded_text, input_names=["x"], output_names=["logits"],
                                   dynamic_axes={'x': {1: "length"}}, f=onnx_path)

if __name__ == "__main__":
    export_to_onnx("torch_model.onnx")