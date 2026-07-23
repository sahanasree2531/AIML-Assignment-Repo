import os
import numpy as np
import tensorflow as tf
from onnx2tf import convert

# ---------------------------------------------------
# Step 1 : Check ONNX Model
# ---------------------------------------------------

onnx_model = "model.onnx"

if not os.path.exists(onnx_model):
    raise FileNotFoundError("model.onnx not found!")

print("✓ ONNX model found.")

# ---------------------------------------------------
# Step 2 : Check Calibration Folder
# ---------------------------------------------------

calib_folder = "calib"

if not os.path.exists(calib_folder):
    raise FileNotFoundError("Calibration folder not found!")

files = sorted(os.listdir(calib_folder))

if len(files) == 0:
    raise ValueError("Calibration folder is empty!")

print(f"✓ {len(files)} calibration files found.")

# ---------------------------------------------------
# Step 3 : Validate Calibration Files
# ---------------------------------------------------

for file in files:

    sample = np.load(os.path.join(calib_folder, file))

    if sample.shape != (1, 28, 28):
        raise ValueError(f"{file} has wrong shape {sample.shape}")

    if np.isnan(sample).any():
        raise ValueError(f"{file} contains NaN values")

    if np.isinf(sample).any():
        raise ValueError(f"{file} contains Inf values")

print("✓ Calibration files validated.")

# ---------------------------------------------------
# Step 4 : Convert ONNX -> TensorFlow
# ---------------------------------------------------

print("\nConverting ONNX to TensorFlow SavedModel...\n")

convert(
    input_onnx_file_path="model.onnx",
    output_folder_path="saved_model"
)

print("\n✓ SavedModel created successfully.")

# ---------------------------------------------------
# Step 5 : Representative Dataset
# ---------------------------------------------------

def representative_dataset():

    for file in files:

        sample = np.load(os.path.join(calib_folder, file))

        sample = np.transpose(sample, (1, 2, 0))

        sample = np.expand_dims(sample, axis=0)

        sample = sample.astype(np.float32)

        yield [sample]

# ---------------------------------------------------
# Step 6 : Load SavedModel
# ---------------------------------------------------

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

# ---------------------------------------------------
# Step 7 : INT8 Quantization
# ---------------------------------------------------

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8

converter.inference_output_type = tf.int8

print("\nConverting to Fully INT8 TFLite...\n")

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✓ model_int8.tflite saved.")

# ---------------------------------------------------
# Step 8 : Verify Model
# ---------------------------------------------------

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

print("\nVerification")

print("-"*40)

print("Input dtype :", input_details[0]["dtype"])

print("Output dtype:", output_details[0]["dtype"])

print()

print("Input Quantization :")

print(input_details[0]["quantization"])

print()

print("Output Quantization :")

print(output_details[0]["quantization"])

print()

size = os.path.getsize("model_int8.tflite") / 1024

print(f"Model Size : {size:.2f} KiB")

# ---------------------------------------------------
# Step 9 : Test Inference
# ---------------------------------------------------

sample = np.load(os.path.join(calib_folder, files[0]))

sample = np.transpose(sample, (1, 2, 0))

sample = np.expand_dims(sample, axis=0)

scale, zero = input_details[0]["quantization"]

sample = sample / scale + zero

sample = sample.astype(np.int8)

interpreter.set_tensor(input_details[0]["index"], sample)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

print()

print("Inference Successful!")

print("Output:")

print(output)