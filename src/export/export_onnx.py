import torch
import onnx
from pathlib import Path
from src.models.model import TinyEyeNetV2

CHECKPOINT_PATH = Path("training_output/best_model.pth")
ONNX_PATH = Path("training_output/tinydrowsy_fp32.onnx")
IMAGE_SIZE = 64

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = TinyEyeNetV2(num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

# 1. Trace the model to a TorchScript module
traced_model = torch.jit.trace(model, dummy_input)

# 2. Export using the legacy exporter (dynamo=False)
torch.onnx.export(
    traced_model,                     # traced module
    dummy_input,                      # example input
    str(ONNX_PATH),
    input_names=['input'],
    output_names=['logits'],
    opset_version=11,                 # or 13 – both work with ESP-PPQ
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    dynamo=False,                     # <-- forces the legacy exporter
)

# Validate
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print(f"✅ ONNX model exported to {ONNX_PATH}")
print(f"   Size: {ONNX_PATH.stat().st_size / 1024:.2f} KB")
print(f"   Opset: {onnx_model.opset_import[0].version}")