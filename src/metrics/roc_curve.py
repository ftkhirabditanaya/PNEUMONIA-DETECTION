import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_roc_curve():
    print("Loading data...")
    _, _, test_loader = get_dataloaders(
        data_root="data/raw/chest_xray",
        batch_size=16,
        num_workers=0
    )

    print("Loading trained model...")

    model = DenseNetAttention(num_classes=2)
    
    checkpoint = torch.load("models/phase1_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probs = []

    print("Running inference...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)

            # ✅ FIX 1: Use SOFTMAX probabilities (NOT argmax)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ✅ FIX 2: Ensure both classes exist
    if len(np.unique(all_labels)) < 2:
        print("❌ ROC cannot be computed (only one class present)")
        return

    # ✅ ROC computation
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"✅ AUC Score: {roc_auc:.4f}")

    # ✅ Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")

    # Random baseline
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Pneumonia Detection")
    plt.legend(loc="lower right")
    plt.grid(True)

    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/roc_curve.png")
    plt.show()

    print("📊 ROC curve saved at: outputs/roc_curve.png")


if __name__ == "__main__":
    plot_roc_curve()