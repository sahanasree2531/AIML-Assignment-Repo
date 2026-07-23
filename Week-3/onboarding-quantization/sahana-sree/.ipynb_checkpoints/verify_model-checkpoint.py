import tensorflow as tf
import numpy as np
import os

print("=" * 60)
print("Verifying INT8 Model")
print("=" * 60)

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput Tensor")
print(input_details)

print("\nOutput Tensor")
print(output_details)

print("\nInput dtype:", input_details[0]["dtype"])
print("Output dtype:", output_details[0]["dtype"])

print("\nInput shape:", input_details[0]["shape"])

print("\nModel Size: %.2f KiB" % (os.path.getsize("model_int8.tflite") / 1024))

sample = np.load("calib/0.npy")

print("\nLoaded shape:", sample.shape)

# Convert (1,28,28) -> (28,28)
sample = sample[0]

print("After removing batch:", sample.shape)

# Convert (28,28) -> (28,28,1)
sample = sample[..., np.newaxis]

print("After adding channel:", sample.shape)

# Convert (28,28,1) -> (1,28,28,1)
sample = sample[np.newaxis, ...]

print("Final input:", sample.shape)

scale, zero = input_details[0]["quantization"]

sample = sample.astype(np.float32)

sample = sample / scale + zero

sample = np.clip(sample, -128, 127).astype(np.int8)

interpreter.set_tensor(input_details[0]["index"], sample)

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

print("\nOutput:")
print(output)

print("\nPrediction:", np.argmax(output))