"""
AI600 Assignment 3 - Task 2 Part A: Fine-tuning ResNet-18 on STL-10
- Loads pre-trained ResNet-18
- Freezes backbone (all conv layers)
- Replaces final FC with 10-class head
- Trains only the head on STL-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Data — STL-10 (96×96 RGB, 10 classes)
# ResNet-18 expects 224×224, so we resize.
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\Matee\Desktop\2026\Deep Learning\Assignment 3\Task2_PartA\data"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std =[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

print("\nLoading STL-10 from local files...")
train_set = datasets.STL10(root=DATA_DIR, split="train", download=False, transform=train_transform)
test_set  = datasets.STL10(root=DATA_DIR, split="test",  download=False, transform=test_transform)

print(f"Train samples : {len(train_set)}")
print(f"Test  samples : {len(test_set)}")
print(f"Classes       : {train_set.classes}")

BATCH = 32   # smaller batch — 224×224 images are large
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

# ─────────────────────────────────────────────
# Model — Pre-trained ResNet-18
# ─────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Step 1: Freeze ALL layers
for param in model.parameters():
    param.requires_grad = False

# Step 2: Replace final FC layer with new 10-class head (unfrozen by default)
in_features = model.fc.in_features          # 512 for ResNet-18
model.fc = nn.Linear(in_features, 10)       # new head — requires_grad=True by default

model = model.to(device)

# Verify: only FC params are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters    : {total:,}")
print(f"Trainable parameters: {trainable:,}  (head only)")

# ─────────────────────────────────────────────
# Training & Evaluation Functions
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
EPOCHS = 15
LR     = 1e-3

# Only pass trainable parameters to optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
criterion = nn.CrossEntropyLoss()

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc * 100)
    history["val_acc"].append(vl_acc * 100)

    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
          f"Test  Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%")

# Final evaluation
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n{'='*55}")
print(f"Final Test Accuracy : {test_acc*100:.2f}%")
print(f"Final Test Loss     : {test_loss:.4f}")
print(f"{'='*55}")

# ─────────────────────────────────────────────
# Plot Training Curves
# ─────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Test Loss",  marker="s")
axes[0].set_title("Loss Curves — ResNet-18 on STL-10")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Test Acc",  marker="s")
axes[1].set_title("Accuracy Curves — ResNet-18 on STL-10")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f"ResNet-18 Fine-tuning (Head Only)  |  Test Acc: {test_acc*100:.2f}%", fontsize=12)
plt.tight_layout()
plt.savefig("stl10_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: stl10_training_curves.png")

# Save model for Part B (GradCAM)
torch.save(model.state_dict(), "resnet18_stl10.pth")
print("Model saved: resnet18_stl10.pth")