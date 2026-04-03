import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from Model.attention_resnet18 import AttentionResNet18
from PIL import Image
import numpy as np
import time

# ------------------------------------------------------------------
# Dataset — swap this out for your ISIC loader
# ------------------------------------------------------------------
class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((135, 135)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((135, 135)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------
def train(n_classes=2, epochs=30, batch_size=32, lr=1e-4, dropout=0.4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- swap these with your actual file paths and labels ---
    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    train_dataset = SkinDataset(train_paths, train_labels, transform=train_transform)
    val_dataset   = SkinDataset(val_paths,   val_labels,   transform=val_transform)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    model     = AttentionResNet18(n_classes=n_classes, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()   # TODO: swap for FocalLoss for ISIC imbalance

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        start = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        # --- validate ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {correct/total:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_correct/val_total:.4f} | "
              f"Time: {(time.time()-start)/60:.1f}min")

    torch.save(model.state_dict(), 'attention_resnet18.pth')
    return model

if __name__ == '__main__':
    train()