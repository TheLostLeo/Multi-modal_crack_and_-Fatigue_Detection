import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

TRAIN_DATA = "data/processed/ae/X_train.npy"
TRAIN_LABELS = "data/processed/ae/y_train.npy"
TEST_DATA = "data/processed/ae/X_test.npy"
TEST_LABELS = "data/processed/ae/y_test.npy"

CHECKPOINT_DIR = "checkpoints/audio"
LOG_DIR = "logs/audio"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Using device:", DEVICE)


# ==========================
# DATASET
# ==========================

class AudioDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def spec_augment(self, x):
        # simple time masking
        if np.random.rand() < 0.5:
            t = np.random.randint(0, x.shape[1] // 4)
            t0 = np.random.randint(0, x.shape[1] - t)
            x[:, t0:t0+t] = 0
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = self.spec_augment(x.copy())

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


# ==========================
# MODEL (Conv2D)
# ==========================

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==========================
# TRAIN FUNCTION
# ==========================

def train():

    X_train = np.load(TRAIN_DATA)
    y_train = np.load(TRAIN_LABELS)
    X_test = np.load(TEST_DATA)
    y_test = np.load(TEST_LABELS)

    train_dataset = AudioDataset(X_train, y_train, augment=True)
    test_dataset = AudioDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print("Train Loss:", round(train_loss, 4))
        print("Val Accuracy:", round(acc, 4))
        print("Val Macro-F1:", round(f1, 4))

        scheduler.step(train_loss)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0

            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best_model.pth"))

            cm = confusion_matrix(all_labels, all_preds)
            np.save(os.path.join(LOG_DIR, "confusion_matrix.npy"), cm)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nBest Validation F1:", best_f1)


if __name__ == "__main__":
    train()

