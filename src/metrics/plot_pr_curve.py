import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path

from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_pr_curve():
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

            # IMPORTANT: probability for PNEUMONIA class
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)

    print(f" Average Precision (AP): {ap_score:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR Curve (AP = {ap_score:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Pneumonia Detection")
    plt.legend(loc="lower left")
    plt.grid(True)

    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/pr_curve.png")
    plt.show()

    print("PR curve saved at: outputs/pr_curve.png")


if __name__ == "__main__":
    plot_pr_curve()