# ==========================================================
# Generate Calibration Data for INT8 Quantization
# ==========================================================

import os
import numpy as np
from torchvision import datasets, transforms

# ----------------------------------------------------------
# Load MNIST Test Dataset
# ----------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# ----------------------------------------------------------
# Create Calibration Folder
# ----------------------------------------------------------

os.makedirs("calib", exist_ok=True)

# ----------------------------------------------------------
# Save First 50 Test Images as .npy Files
# ----------------------------------------------------------

num_samples = 50

for i in range(num_samples):

    image, label = test_data[i]

    np.save(f"calib/{i}.npy", image.numpy())

print(f"\nSuccessfully saved {num_samples} calibration samples.")

print("\nCalibration folder: calib/")