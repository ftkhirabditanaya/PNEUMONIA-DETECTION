import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():

    _, _, test_loader = get_dataloaders("data/raw/chest_xray")

    model = DenseNetAttention().to(device)
    model.load_state_dict(torch.load("models/densenet.pth"))  # we'll save next

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:,1]

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nROC-AUC Score:")
    print(roc_auc_score(all_labels, all_probs))


if __name__ == "__main__":
    evaluate()