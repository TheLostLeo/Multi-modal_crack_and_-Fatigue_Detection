import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.datasets.image_dataset import ImageDataset
from src.models.image_model import get_model
from src.utils.metrics import compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):

    best_f1 = 0
    best_cm = None

    for epoch in range(epochs):

        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        model.train()
        running_loss = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for x, y in train_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.set_postfix(
                loss=running_loss / (train_bar.n + 1)
            )

        model.eval()
        all_preds = []
        all_labels = []

        val_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for x, y in val_bar:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc, macro_f1, cm = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Val Accuracy: {acc:.4f}")
        print(f"Val Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_cm = cm

    return best_f1, best_cm


def main():

    config = load_config("config/image_config.yaml")
    device = torch.device(config["training"]["device"])

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    train_dataset = ImageDataset(config["paths"]["train_csv"], train=True)
    val_dataset = ImageDataset(config["paths"]["val_csv"], train=False)


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = get_model(**config["model"]).to(device)

    labels = train_dataset.df["label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    best_f1, best_cm = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        config["training"]["epochs"]
    )

    torch.save(
        model.state_dict(),
        os.path.join(config["paths"]["checkpoint_dir"], "best_model.pth")
    )

    log_dir = config["paths"]["log_dir"]

    np.save(os.path.join(log_dir, "confusion_matrix.npy"), best_cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Image Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    plt.close()

    print("\nBest Validation F1:", best_f1)


if __name__ == "__main__":
    main()

