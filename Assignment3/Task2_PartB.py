"""
AI600 Assignment 3 - Task 2 Part B: GradCAM Visualization
Loads fine-tuned ResNet-18, runs GradCAM on 4 STL-10 test images:
  - 2 correctly classified
  - 2 incorrectly classified
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Load STL-10 test set
# ─────────────────────────────────────────────
DATA_DIR  = r"C:\Users\Matee\Desktop\2026\Deep Learning\Assignment 3\Task2_PartA\data"
MODEL_PATH = r"C:\Users\Matee\Desktop\2026\Deep Learning\Assignment 3\Task2_PartA\resnet18_stl10.pth"
OUT_DIR    = r"C:\Users\Matee\Desktop\2026\Deep Learning\Assignment 3\Task2_PartA"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Unnormalize for display
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std =[1/0.229,       1/0.224,       1/0.225]
)

test_set    = datasets.STL10(root=DATA_DIR, split="test", download=False, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
class_names = test_set.classes
print(f"Classes: {class_names}")

# ─────────────────────────────────────────────
# Load fine-tuned ResNet-18
# ─────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded.")

# ─────────────────────────────────────────────
# Find 2 correct + 2 incorrect predictions
# ─────────────────────────────────────────────
correct_samples   = []
incorrect_samples = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1)

        for i in range(len(imgs)):
            if preds[i] == labels[i] and len(correct_samples) < 2:
                correct_samples.append((imgs[i].cpu(), labels[i].item(), preds[i].item()))
            elif preds[i] != labels[i] and len(incorrect_samples) < 2:
                incorrect_samples.append((imgs[i].cpu(), labels[i].item(), preds[i].item()))

        if len(correct_samples) == 2 and len(incorrect_samples) == 2:
            break

samples = correct_samples + incorrect_samples
labels_list = (["Correct"] * 2) + (["Incorrect"] * 2)
print(f"Found {len(correct_samples)} correct and {len(incorrect_samples)} incorrect samples.")

# ─────────────────────────────────────────────
# GradCAM Implementation
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, class_idx=None):
        """
        img_tensor: (1, C, H, W) on device
        Returns: heatmap as numpy array (H, W) in [0,1]
        """
        self.model.zero_grad()
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop w.r.t. predicted class score
        score = output[0, class_idx]
        score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# Target layer: last conv layer of ResNet-18
gradcam = GradCAM(model, model.layer4[-1])

# ─────────────────────────────────────────────
# Generate heatmaps and plot
# ─────────────────────────────────────────────
def to_display(img_tensor):
    """Unnormalize and convert to HWC numpy for display."""
    img = inv_normalize(img_tensor.clone())
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
titles = ["Original Image", "GradCAM Heatmap", "Overlay"]

for row, ((img, true_label, pred_label), result) in enumerate(zip(samples, labels_list)):
    img_input = img.unsqueeze(0).to(device)
    img_input.requires_grad_(False)

    # Generate GradCAM
    heatmap = gradcam.generate(img_input, class_idx=pred_label)

    # Resize heatmap to 224x224
    heatmap_resized = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = torch.nn.functional.interpolate(
        heatmap_resized, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Original image for display
    img_display = to_display(img)

    # Colormap heatmap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    # Overlay
    overlay = 0.5 * img_display + 0.5 * heatmap_colored
    overlay = overlay.clip(0, 1)

    # Status color
    color = "green" if result == "Correct" else "red"

    # Plot
    axes[row, 0].imshow(img_display)
    axes[row, 0].set_title(
        f"True: {class_names[true_label]}\nPred: {class_names[pred_label]} [{result}]",
        fontsize=9, color=color, fontweight="bold"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(heatmap_resized, cmap="jet")
    axes[row, 1].set_title("GradCAM Heatmap", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("Overlay", fontsize=9)
    axes[row, 2].axis("off")

# Column headers
for col, title in enumerate(titles):
    axes[0, col].set_title(
        title + "\n" + axes[0, col].get_title(),
        fontsize=9, color="green", fontweight="bold"
    )

plt.suptitle("GradCAM Visualization — Fine-tuned ResNet-18 on STL-10\n"
             "Rows 1-2: Correct Predictions  |  Rows 3-4: Incorrect Predictions",
             fontsize=12, fontweight="bold")
plt.tight_layout()

save_path = OUT_DIR + r"\gradcam_results.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {save_path}")