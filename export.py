import torch

def export_to_onnx(onnx_path):
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    torch_model = torch.load("models/gpt_trained_17000.pth")
    torch_input = torch.randn(1, 500)
    onnx_model = torch.onnx.dynamo_export(torch_model, torch_input, export_options=export_options)
    onnx_model.save(onnx_path)

if __name__ == "__main__":
    export_to_onnx("torch_model.onnx")