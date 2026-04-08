"""
AI600 Assignment 3 - Task 1 Part A
Training & Evaluation Code for Custom CNN on Standard MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Model Definition (≤3 conv, ≤2 FC, ≤50k params)
# ─────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # Conv1
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28 → 14

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14 → 7

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Conv3
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 7 → 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),                   # FC1
            nn.ReLU(),
            nn.Linear(64, 10),                            # FC2 → 10 classes
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# Data Loading & Splits
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean & std
])

full_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 90/10 train-validation split
val_size   = int(0.1 * len(full_train))   # 6,000
train_size = len(full_train) - val_size   # 54,000
train_set, val_set = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

BATCH = 128
# num_workers=0 required on Windows (avoids multiprocessing spawn error)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)


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

model     = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")
assert total_params <= 50_000, f"Parameter limit exceeded! ({total_params:,})"

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc * 100)
    history["val_acc"].append(vl_acc * 100)

    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
          f"Val   Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%")

# Final test evaluation
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n{'='*55}")
print(f"Final Test Accuracy : {test_acc*100:.2f}%")
print(f"Final Test Loss     : {test_loss:.4f}")
print(f"{'='*55}")


# ─────────────────────────────────────────────
# Plot Training Curves (for Q1.1)
# ─────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="s")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",   marker="s")
axes[1].set_title("Accuracy Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f"TinyCNN on MNIST  |  Test Acc: {test_acc*100:.2f}%", fontsize=13)
plt.tight_layout()
plt.savefig("mnist_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: mnist_training_curves.png")


# ─────────────────────────────────────────────
# Visualize Conv1 Filters (for Q1.2)
# ─────────────────────────────────────────────
conv1_weights = model.features[0].weight.detach().cpu()  # (8, 1, 3, 3)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    filt = conv1_weights[i, 0].numpy()
    filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
    ax.imshow(filt_norm, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title(f"Filter {i+1}", fontsize=9)
    ax.axis("off")

plt.suptitle("Conv1 Filters (3×3, 8 filters) — Red=positive, Blue=negative", fontsize=11)
plt.tight_layout()
plt.savefig("conv1_filters.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: conv1_filters.png")

# Save model checkpoint
torch.save(model.state_dict(), "tinycnn_mnist.pth")
print("Model saved: tinycnn_mnist.pth")
