"""
AI600 Assignment 3 - Task 1 Part B: Colored-MNIST (C-MNIST)
Loads .pt files, trains 3-channel CNN, evaluates on biased & unbiased test sets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Load .pt files
# Each .pt file is expected to contain a dict with keys 'images' and 'labels'
# or a tuple (images, labels). We handle both cases below.
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\Matee\Desktop\2026\Deep Learning\Assignment 3\cmnist"

def load_pt(filename):
    path = f"{DATA_DIR}\\{filename}"
    data = torch.load(path)
    # Handle both dict and tuple formats
    if isinstance(data, dict):
        images = data['images'].float()
        labels = data['labels'].long()
    elif isinstance(data, (list, tuple)):
        images = data[0].float()
        labels = data[1].long()
    else:
        raise ValueError(f"Unknown format in {filename}: {type(data)}")

    # Normalize to [0, 1] if images are in [0, 255]
    if images.max() > 1.0:
        images = images / 255.0

    # Ensure shape is (N, C, H, W)
    if images.ndim == 3:          # (N, H, W) — grayscale, add channel dim
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[-1] == 3:  # (N, H, W, 3) → (N, 3, H, W)
        images = images.permute(0, 3, 1, 2)

    print(f"  {filename}: images={tuple(images.shape)}, labels={tuple(labels.shape)}, "
          f"classes={labels.unique().numel()}")
    return images, labels

print("\nLoading datasets...")
train_imgs, train_lbls   = load_pt("train_biased.pt")
test_b_imgs, test_b_lbls = load_pt("test_biased.pt")
test_u_imgs, test_u_lbls = load_pt("test_unbiased.pt")

# Detect number of channels from data
in_channels = train_imgs.shape[1]
print(f"\nInput channels detected: {in_channels}")

# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────
BATCH = 128

train_loader    = DataLoader(TensorDataset(train_imgs, train_lbls),
                             batch_size=BATCH, shuffle=True)
test_b_loader   = DataLoader(TensorDataset(test_b_imgs, test_b_lbls),
                             batch_size=BATCH, shuffle=False)
test_u_loader   = DataLoader(TensorDataset(test_u_imgs, test_u_lbls),
                             batch_size=BATCH, shuffle=False)

# ─────────────────────────────────────────────
# Model — same TinyCNN but with configurable input channels
# ─────────────────────────────────────────────
class TinyCNN_RGB(nn.Module):
    """
    Same architecture as Part A TinyCNN, but accepts in_channels (3 for RGB).
    All other constraints preserved: 3 conv, 2 FC, <50k params.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 28→14

            nn.Conv2d(8, 16, kernel_size=3, padding=1),            # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 14→7

            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # Conv3
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # 7→3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),                             # FC1
            nn.ReLU(),
            nn.Linear(64, 10),                                      # FC2
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
# Training
# ─────────────────────────────────────────────
EPOCHS = 15
LR     = 1e-3

model     = TinyCNN_RGB(in_channels=in_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

history = {"train_loss": [], "train_acc": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc * 100)

    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%")

# ─────────────────────────────────────────────
# Final Evaluation on both test sets
# ─────────────────────────────────────────────
_, biased_acc   = evaluate(model, test_b_loader, criterion)
_, unbiased_acc = evaluate(model, test_u_loader, criterion)

print(f"\n{'='*55}")
print(f"Biased   Test Accuracy : {biased_acc*100:.2f}%")
print(f"Unbiased Test Accuracy : {unbiased_acc*100:.2f}%")
print(f"Accuracy Drop          : {(biased_acc - unbiased_acc)*100:.2f}%")
print(f"{'='*55}")

# ─────────────────────────────────────────────
# Plot Training Curves
# ─────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="steelblue")
axes[0].set_title("Training Loss (C-MNIST)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o", color="darkorange")
axes[1].set_title("Training Accuracy (C-MNIST)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(
    f"TinyCNN-RGB on C-MNIST  |  "
    f"Biased: {biased_acc*100:.2f}%  |  Unbiased: {unbiased_acc*100:.2f}%",
    fontsize=12
)
plt.tight_layout()
plt.savefig("cmnist_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: cmnist_training_curves.png")

# ─────────────────────────────────────────────
# Visualize sample images from each test set
# ─────────────────────────────────────────────
def show_samples(imgs, lbls, title, filename, n=10):
    fig, axes = plt.subplots(1, n, figsize=(14, 2))
    for i in range(n):
        img = imgs[i].permute(1, 2, 0).numpy()   # (C,H,W) → (H,W,C)
        img = img.clip(0, 1)
        axes[i].imshow(img if img.shape[2] == 3 else img[:,:,0], 
                       cmap=None if img.shape[2] == 3 else "gray")
        axes[i].set_title(str(lbls[i].item()), fontsize=9)
        axes[i].axis("off")
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

show_samples(test_b_imgs, test_b_lbls,
             "Biased Test Set Samples",   "cmnist_biased_samples.png")
show_samples(test_u_imgs, test_u_lbls,
             "Unbiased Test Set Samples", "cmnist_unbiased_samples.png")

torch.save(model.state_dict(), "tinycnn_rgb_cmnist.pth")
print("Model saved: tinycnn_rgb_cmnist.pth")
