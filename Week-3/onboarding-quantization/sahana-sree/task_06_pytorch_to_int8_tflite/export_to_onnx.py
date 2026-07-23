# ==========================================================
# Export PyTorch Model to ONNX
# ==========================================================

import torch
import onnx
from model_definition import SimpleCNN

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully.")

# ----------------------------------------------------------
# Create Dummy Input
# ----------------------------------------------------------

dummy_input = torch.randn(1, 1, 28, 28)

# ----------------------------------------------------------
# Export to ONNX
# ----------------------------------------------------------

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes=None
)

print("Model exported to model.onnx")

# ----------------------------------------------------------
# Verify ONNX Model
# ----------------------------------------------------------

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model verification successful.")