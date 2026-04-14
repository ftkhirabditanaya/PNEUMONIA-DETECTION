from src.data_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders("data/raw/chest_xray")

for images, labels in train_loader:
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    break