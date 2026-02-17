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
# ACD WAVEFORM INSPECTION
# ==========================

def inspect_acd_waveforms():

    print("\n=== ACD WAVEFORM DATA ===")

    X_train = np.load(DATA_ROOT / "processed/ae/waveforms_train.npy")
    y_train = np.load(DATA_ROOT / "processed/ae/labels_train.npy")

    X_test = np.load(DATA_ROOT / "processed/ae/waveforms_test.npy")
    y_test = np.load(DATA_ROOT / "processed/ae/labels_test.npy")

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nTrain Label Distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nTest Label Distribution:")
    print(pd.Series(y_test).value_counts())

    print("\nWaveform statistics (train):")
    print("Mean:", X_train.mean())
    print("Std:", X_train.std())
    print("Min:", X_train.min())
    print("Max:", X_train.max())

    # Visualize random waveform
    idx = random.randint(0, len(X_train) - 1)
    waveform = X_train[idx]
    label = y_train[idx]

    print(f"\nVisualizing waveform index {idx}, Label {label}")

    plt.figure(figsize=(10, 4))
    plt.plot(waveform)
    plt.title(f"ACD Waveform (Label {label})")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (mV)")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "acd_waveform_sample.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved waveform plot to: {save_path}")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    inspect_images()
    inspect_acd_waveforms()
