import torch
from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention
from src.losses import FocalLoss

from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():

    print("Loading data...")
    _, _, test_loader = get_dataloaders(
        data_root="data/raw/chest_xray",
        batch_size=16,
        num_workers=0,
        val_split=0.2,
        seed=42
    )

    print("Loading best model...")
    model = DenseNetAttention(
        num_classes=2,
        dropout_rate=0.4,
        freeze_backbone=False
    ).to(DEVICE)

    checkpoint = torch.load(
    "models/phase1_best.pth",
    map_location=DEVICE,
    weights_only=False   # 🔥 FIX
)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    test()