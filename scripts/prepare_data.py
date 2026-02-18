import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy import signal
from sklearn.model_selection import train_test_split


# ==========================
# GLOBAL SETTINGS
# ==========================

IMAGE_SIZE = 224
SPECTROGRAM_NFFT = 256
SPECTROGRAM_HOP = 128
TARGET_TIME_BINS = 64
IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]


# ==========================
# UTILITIES
# ==========================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


# ==========================
# IMAGE PROCESSING
# ==========================

def process_images(raw_dir, processed_dir):

    print("Processing SDNET2018 images...")

    image_paths = []
    labels = []

    for root, _, files in os.walk(raw_dir):
        for file in files:
            if is_image_file(file):
                full_path = os.path.join(root, file)
                folder_name = Path(root).name.upper()
                label = 1 if folder_name.startswith("C") else 0
                image_paths.append(full_path)
                labels.append(label)

    df = pd.DataFrame({"filepath": image_paths, "label": labels})

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=42
    )

    ensure_dir(processed_dir)

    def save_split(split_df, split_name):
        split_path = Path(processed_dir) / split_name
        ensure_dir(split_path)

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img = cv2.imread(row["filepath"])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            save_path = split_path / f"{Path(row['filepath']).stem}.jpg"
            cv2.imwrite(str(save_path), img)

    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")

    print("Image preprocessing complete.")
    print("Total samples:", len(df))


# ==========================
# AUDIO PROCESSING
# ==========================

def waveform_to_spectrogram(waveform):

    if np.all(waveform == 0):
        return None

    try:
        f, t, spec = signal.spectrogram(
            waveform,
            nperseg=SPECTROGRAM_NFFT,
            noverlap=SPECTROGRAM_NFFT - SPECTROGRAM_HOP,
            scaling="spectrum",
            mode="magnitude"
        )

        if spec.size == 0:
            return None

        spec = np.log(spec + 1e-8)

        if np.isnan(spec).any() or np.isinf(spec).any():
            return None

        mean = np.mean(spec)
        std = np.std(spec)

        if std < 1e-6:
            return None

        spec = (spec - mean) / (std + 1e-6)

        time_bins = spec.shape[1]

        if time_bins < TARGET_TIME_BINS:
            pad_width = TARGET_TIME_BINS - time_bins
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
        elif time_bins > TARGET_TIME_BINS:
            spec = spec[:, :TARGET_TIME_BINS]

        return spec.astype(np.float32)

    except Exception:
        return None


def process_audio(raw_dir, processed_dir):

    print("Processing ACD waveform dataset...")

    train_path = Path(raw_dir) / "acd_model_data_5050_sm_df.pkl"
    test_path = Path(raw_dir) / "acd_test_data_df.pkl"

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    print("\nOriginal train channel distribution:")
    print(train_df["channel"].value_counts())

    print("\nTest channel distribution:")
    print(test_df["channel"].value_counts())

    # ---- Safe channel balancing ----
    print("\nBalancing training channels...")

    min_count = train_df["channel"].value_counts().min()

    balanced_parts = []
    for ch in train_df["channel"].unique():
        subset = train_df[train_df["channel"] == ch]
        subset = subset.sample(min_count, random_state=42)
        balanced_parts.append(subset)

    balanced_df = pd.concat(balanced_parts).reset_index(drop=True)

    print("\nBalanced train channel distribution:")
    print(balanced_df["channel"].value_counts())

    # ---- Generate spectrograms ----

    X_train, y_train = [], []

    print("\nGenerating train spectrograms...")
    for _, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        waveform = np.array(row["waveform_noz"], dtype=np.float32)
        spec = waveform_to_spectrogram(waveform)

        if spec is not None:
            X_train.append(spec)
            y_train.append(int(row["crack"]))

    X_test, y_test = [], []

    print("\nGenerating test spectrograms...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        waveform = np.array(row["waveform_noz"], dtype=np.float32)
        spec = waveform_to_spectrogram(waveform)

        if spec is not None:
            X_test.append(spec)
            y_test.append(int(row["crack"]))

    X_train = np.stack(X_train)
    y_train = np.array(y_train)

    X_test = np.stack(X_test)
    y_test = np.array(y_test)

    ensure_dir(processed_dir)

    np.save(Path(processed_dir) / "X_train.npy", X_train)
    np.save(Path(processed_dir) / "y_train.npy", y_train)
    np.save(Path(processed_dir) / "X_test.npy", X_test)
    np.save(Path(processed_dir) / "y_test.npy", y_test)

    print("\nAudio preprocessing complete.")
    print("Train spectrogram shape:", X_train.shape)
    print("Test spectrogram shape:", X_test.shape)


# ==========================
# MAIN
# ==========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    raw_images = data_root / "raw/images/SDNET2018"
    raw_audio = data_root / "raw/ae"

    processed_images = data_root / "processed/images"
    processed_audio = data_root / "processed/ae"

    process_images(raw_images, processed_images)
    process_audio(raw_audio, processed_audio)


if __name__ == "__main__":
    main()

