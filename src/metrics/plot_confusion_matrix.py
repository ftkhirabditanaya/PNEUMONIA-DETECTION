import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion():
    print("Loading data...")
    _, _, test_loader = get_dataloaders(
        data_root="data/raw/chest_xray",
        batch_size=16,
        num_workers=0
    )

    print("Loading trained model...")
    model = DenseNetAttention(num_classes=2)

    #  FIX: load correct checkpoint safely
    checkpoint = torch.load("models/phase1_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    print("Running inference...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    classes = ["NORMAL", "PNEUMONIA"]

    # Convert to percentage
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Professional heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_percent,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Counts + %)")
    plt.tight_layout()

    # Save output
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.show()

    print(" Saved at: outputs/confusion_matrix.png")


if __name__ == "__main__":
    plot_confusion()