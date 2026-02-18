import os
import cv2
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ===============================
# GLOBAL CONFIG
# ===============================

DATA_ROOT = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
OUTPUT_DIR = Path("output")

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# DATASET
# ===============================

class CrackDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        label = int(self.df.iloc[idx]["label"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label


# ===============================
# AUGMENTATION
# ===============================

train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ===============================
# MODEL (Classification + Segmentation-ready)
# ===============================

class CrackModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        feature_maps = self.features(x)          # (B, 512, H, W)
        pooled = self.pool(feature_maps)         # (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)
        return logits, feature_maps


# ===============================
# GRAD-CAM
# ===============================

def generate_gradcam(model, image_tensor, save_path):

    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    image_tensor.requires_grad = True

    logits, feature_maps = model(image_tensor)
    pred_class = torch.argmax(logits, dim=1)

    model.zero_grad()
    logits[:, pred_class].backward()

    gradients = image_tensor.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activation = feature_maps.detach().cpu()[0]
    for i in range(activation.shape[0]):
        activation[i] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    plt.imshow(heatmap, cmap='jet')
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


# ===============================
# TRAINING LOOP
# ===============================

def train_model(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_model = copy.deepcopy(model.state_dict())
    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        # ---- Training ----
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs, _ = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {acc:.4f}")
        print(f"Val Macro-F1: {f1:.4f}")

        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        np.save(LOG_DIR / "confusion_matrix.npy", cm)

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "image_model.pth")

    print("\nBest Validation F1:", best_f1)

    return model


# ===============================
# MAIN
# ===============================

def main():

    print("Using device:", DEVICE)

    train_csv = DATA_ROOT / "metadata/image_train.csv"
    val_csv = DATA_ROOT / "metadata/image_val.csv"

    train_dataset = CrackDataset(train_csv, transform=train_transform)
    val_dataset = CrackDataset(val_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CrackModel().to(DEVICE)

    model = train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

