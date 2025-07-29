import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

# Load hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and loaders
train_data = datasets.ImageFolder("data/processed/split/train", transform=train_transform)
val_data = datasets.ImageFolder("data/processed/split/val", transform=val_test_transform)
test_data = datasets.ImageFolder("data/processed/split/test", transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=params["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=params["train"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_data, batch_size=params["train"]["batch_size"], shuffle=False)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, params["train"]["num_classes"])
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params["train"]["lr"])

# Early stopping parameters
patience = params["train"].get("patience", 5)
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(params["train"]["epochs"]):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss:   {avg_val_loss:.4f}, Val Accuracy:   {val_accuracy:.2f}%")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "models/resnet18_best.pth")
        print("✅ Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement. Early stop counter: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
        print("⛔ Early stopping triggered.")
        break

# Final model save
torch.save(model.state_dict(), "models/resnet18_final.pth")

# Test set evaluation
model.load_state_dict(torch.load("models/resnet18_best.pth"))
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_accuracy = 100 * correct_test / total_test
print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")

