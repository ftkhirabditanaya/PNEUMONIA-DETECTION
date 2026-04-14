import os
import cv2
import torch
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        classes = ["NORMAL", "PNEUMONIA"]

        for label, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)

            images = os.listdir(cls_path)

            # 🔥 LIMIT DATA (for CPU debugging)
            if limit:
                images = images[:limit]

            for img in images:
                self.image_paths.append(os.path.join(cls_path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)

        # 🔥 HANDLE CORRUPTED IMAGES
        if image is None:
            return self.__getitem__((idx + 1) % len(self.image_paths))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)