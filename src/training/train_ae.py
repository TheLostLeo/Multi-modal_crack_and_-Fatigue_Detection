import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from src.datasets.ae_dataset import AEDataset
from src.models.ae_model import AEStageClassifier
from src.utils.metrics import compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_fold(model, train_loader, val_loader, optimizer, criterion, device, epochs):

    best_f1 = 0
    best_cm = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc, macro_f1, cm = compute_metrics(all_labels, all_preds)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_cm = cm

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val F1={macro_f1:.4f}")

    return best_f1, best_cm


def main():

    config = load_config("config/ae_config.yaml")

    device = torch.device(config["training"]["device"])

    sequences = np.load(config["paths"]["data_sequences"])
    labels = np.load(config["paths"]["data_labels"])
    groups = np.load(config["paths"]["data_groups"])

    gkf = GroupKFold(n_splits=config["training"]["k_folds"])

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(sequences, labels, groups)
    ):

        print(f"\n===== Fold {fold+1} =====")

        train_dataset = AEDataset(sequences[train_idx], labels[train_idx])
        val_dataset = AEDataset(sequences[val_idx], labels[val_idx])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False
        )

        model = AEStageClassifier(**config["model"]).to(device)

        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        best_f1, best_cm = train_fold(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            config["training"]["epochs"]
        )

        fold_results.append(best_f1)

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(
                config["paths"]["checkpoint_dir"],
                f"group_fold_{fold+1}.pth"
            )
        )

        # Save confusion matrix
        fold_log_dir = os.path.join(config["paths"]["log_dir"], f"fold_{fold+1}")
        os.makedirs(fold_log_dir, exist_ok=True)

        np.save(os.path.join(fold_log_dir, "confusion_matrix.npy"), best_cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(fold_log_dir, "confusion_matrix.png"))
        plt.close()

    print("\n=== Group Cross Validation Results ===")
    print("F1 scores per fold:", fold_results)
    print("Mean F1:", np.mean(fold_results))
    print("Std F1:", np.std(fold_results))


if __name__ == "__main__":
    main()

