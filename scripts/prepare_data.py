import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split


# ==========================
# GLOBAL SETTINGS
# ==========================

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]

TARGET_LENGTH = 4096
RANDOM_STATE = 42


# ==========================
# UTILITIES
# ==========================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def pad_or_truncate(waveform, target_len=TARGET_LENGTH):
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    elif len(waveform) > target_len:
        waveform = waveform[:target_len]
    return waveform


def normalize_waveform(waveform):
    mean = np.mean(waveform)
    std = np.std(waveform)
    if std < 1e-8:
        return None
    return (waveform - mean) / (std + 1e-8)


# ==========================
# IMAGE PREPROCESSING
# ==========================

def process_images(raw_dir, processed_dir, metadata_dir):

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

    df = pd.DataFrame({
        "filepath": image_paths,
        "label": labels
    })

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )

    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=RANDOM_STATE
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

    ensure_dir(metadata_dir)
    train_df.to_csv(Path(metadata_dir) / "image_train.csv", index=False)
    val_df.to_csv(Path(metadata_dir) / "image_val.csv", index=False)
    test_df.to_csv(Path(metadata_dir) / "image_test.csv", index=False)

    print("Image preprocessing complete.")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))


# ==========================
# AUDIO PREPROCESSING (CLEAN BASELINE)
# ==========================

def process_audio(raw_dir, processed_dir):

    print("\nProcessing ACD waveform dataset...")

    train_path = Path(raw_dir) / "acd_model_data_5050_sm_df.pkl"
    test_path = Path(raw_dir) / "acd_test_data_df.pkl"

    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)

    df = pd.concat([df_train, df_test], ignore_index=True)

    print("Original total samples:", len(df))
    print("Original channel distribution:")
    print(df["channel"].value_counts())

    # -------------------------
    # Filter channel 3 only
    # -------------------------
    print("\nFiltering channel 3 only...")
    df = df[df["channel"] == 3.0].reset_index(drop=True)

    print("Filtered samples:", len(df))
    print("Crack distribution after filtering:")
    print(df["crack"].value_counts())

    # -------------------------
    # Build dataset
    # -------------------------
    X = []
    y = []

    print("\nProcessing waveforms...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        waveform = np.array(row["waveform_noz"], dtype=np.float32)

        waveform = normalize_waveform(waveform)
        if waveform is None:
            continue

        waveform = pad_or_truncate(waveform)

        X.append(waveform)
        y.append(int(row["crack"]))

    X = np.stack(X)
    y = np.array(y)

    print("\nFinal dataset shape:", X.shape)
    print("Final crack distribution:", np.bincount(y))

    # -------------------------
    # Stratified split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train crack distribution:", np.bincount(y_train))
    print("Test crack distribution:", np.bincount(y_test))

    ensure_dir(processed_dir)

    np.save(Path(processed_dir) / "X_train.npy", X_train)
    np.save(Path(processed_dir) / "y_train.npy", y_train)
    np.save(Path(processed_dir) / "X_test.npy", X_test)
    np.save(Path(processed_dir) / "y_test.npy", y_test)

    print("\nAudio preprocessing complete.")


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
    metadata_dir = data_root / "metadata"

    process_images(raw_images, processed_images, metadata_dir)
    process_audio(raw_audio, processed_audio)


if __name__ == "__main__":
    main()

