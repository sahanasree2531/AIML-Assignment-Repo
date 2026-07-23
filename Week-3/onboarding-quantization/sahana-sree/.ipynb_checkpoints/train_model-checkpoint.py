# ==========================================================
# Train SimpleCNN on MNIST and Save model.pth
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_definition import SimpleCNN

# ----------------------------------------------------------
# Device Configuration
# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ----------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------

batch_size = 64
learning_rate = 0.001
epochs = 5

# ----------------------------------------------------------
# Load MNIST Dataset
# ----------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

model = SimpleCNN().to(device)

# ----------------------------------------------------------
# Loss Function
# ----------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# ----------------------------------------------------------
# Training Loop
# ----------------------------------------------------------

print("\nTraining Started...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

print("\nTraining Completed!")

# ----------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

torch.save(model.state_dict(), "model.pth")

print("\nModel saved successfully as model.pth")