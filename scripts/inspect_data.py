import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random


DATA_ROOT = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# IMAGE INSPECTION
# ==========================

def inspect_images():

    print("\n=== IMAGE DATA ===")

    train_csv = DATA_ROOT / "metadata/image_train.csv"
    df = pd.read_csv(train_csv)

    print("\nLabel Distribution:")
    print(df["label"].value_counts())

    print("\nTotal training samples:", len(df))

    sample_df = df.sample(4, random_state=42)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img = cv2.imread(row["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Label: {row['label']}")
        ax.axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "image_samples.png"
    plt.savefig(save_path)
    plt.close()

    print(f"\nSaved image samples to: {save_path}")


# ==========================
# AE INSPECTION
# ==========================

def inspect_ae():

    print("\n=== AE DATA ===")

    sequences = np.load(DATA_ROOT / "processed/ae/ae_sequences.npy")
    labels = np.load(DATA_ROOT / "processed/ae/ae_labels.npy")

    print("Sequences shape:", sequences.shape)
    print("Labels shape:", labels.shape)

    print("\nDamage Stage Distribution:")
    print(pd.Series(labels).value_counts())

    # Basic statistics
    print("\nFeature Statistics (global):")
    print("Stress_norm mean:", sequences[:, :, 0].mean())
    print("Hit_rate mean:", sequences[:, :, 1].mean())
    print("Hit_acc mean:", sequences[:, :, 2].mean())

    # Pick random sample
    idx = random.randint(0, len(sequences) - 1)

    seq = sequences[idx]
    stage = labels[idx]

    print(f"\nVisualizing AE sequence index {idx}, Stage {stage}")

    stress_norm = seq[:, 0]
    hit_rate = seq[:, 1]
    hit_acc = seq[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    axes[0].plot(stress_norm)
    axes[0].set_title("Normalized Stress")

    axes[1].plot(hit_rate)
    axes[1].set_title("Hit Rate")

    axes[2].plot(hit_acc)
    axes[2].set_title("Hit Acceleration")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "ae_sample.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved AE sample plot to: {save_path}")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    inspect_images()
    inspect_ae()

