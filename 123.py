import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50


class OrangeLineDataset(Dataset):
    """Кастомный датасет для загрузки изображений и масок."""
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.image_filenames[idx].replace(".jpg", ".png"))

        image = torchvision.io.read_image(img_path).float() / 255.0
        mask = torchvision.io.read_image(mask_path)[0]  # Грейскейл маска

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask

# Директории с вашими данными
images_dir = "JPEGImages"
masks_dir = "masks"

# Инициализация датасета и DataLoader
train_dataset = OrangeLineDataset(images_dir, masks_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Загрузка предобученной модели
model = deeplabv3_resnet50(weights=True)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))  # Для 2 классов: линия и фон
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Оптимизатор и функция потерь
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Цикл обучения

model.train()
total_loss = 0

for images, masks in train_loader:
    images, masks = images.to("cuda"), masks.to("cuda")

    optimizer.zero_grad()
    outputs = model(images)['out']
    loss = criterion(outputs, masks.long())
    loss.backward()
    optimizer.step()

    total_loss += loss.item()


# Сохранение модели
torch.save(model.state_dict(), "deeplabv3_orange_line.pth")







































