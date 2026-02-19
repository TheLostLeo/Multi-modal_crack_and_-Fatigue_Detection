import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

DATA_DIR = "data/processed/ae"
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

    def add_noise(self, x):
        noise = np.random.normal(0, 0.01, size=x.shape)
        return x + noise

    def random_shift(self, x):
        shift = np.random.randint(0, 200)
        return np.roll(x, shift)

    def amplitude_scale(self, x):
        scale = np.random.uniform(0.8, 1.2)
        return x * scale

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                x = self.add_noise(x)
            if np.random.rand() < 0.5:
                x = self.random_shift(x)
            if np.random.rand() < 0.5:
                x = self.amplitude_scale(x)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

# ==========================
# MODEL (1D CNN)
# ==========================

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

# ==========================
# TRAINING
# ==========================

def train():

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ----- CLASS WEIGHTS -----
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Class weights:", class_weights)

    train_dataset = AudioDataset(X_train, y_train, augment=True)
    test_dataset = AudioDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

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

        # ----- VALIDATION -----
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

        scheduler.step(f1)

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

    print("\nBest Crack F1:", best_f1)

# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    train()

