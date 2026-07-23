from onnx2tf import convert
import os

print("=" * 60)
print("ONNX -> TensorFlow Conversion")
print("=" * 60)

if not os.path.exists("model.onnx"):
    print("ERROR: model.onnx not found!")
    exit()

try:
    convert(
        input_onnx_file_path="model.onnx",
        output_folder_path="saved_model"
    )

    print("\nTensorFlow SavedModel created successfully!")

except Exception as e:
    print("\nConversion Failed!")
    print(e)