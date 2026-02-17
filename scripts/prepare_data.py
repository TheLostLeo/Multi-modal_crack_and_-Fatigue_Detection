import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


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

def process_images(raw_dir, processed_dir, metadata_dir, resize=224):

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
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=42
    )

    ensure_dir(processed_dir)

    def process_split(split_df, split_name):
        split_path = Path(processed_dir) / split_name
        ensure_dir(split_path)

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img = cv2.imread(row["filepath"])
            img = cv2.resize(img, (resize, resize))
            save_path = split_path / f"{Path(row['filepath']).stem}.jpg"
            cv2.imwrite(str(save_path), img)

    process_split(train_df, "train")
    process_split(val_df, "val")
    process_split(test_df, "test")

    ensure_dir(metadata_dir)
    train_df.to_csv(Path(metadata_dir) / "image_train.csv", index=False)
    val_df.to_csv(Path(metadata_dir) / "image_val.csv", index=False)
    test_df.to_csv(Path(metadata_dir) / "image_test.csv", index=False)

    print("Image processing complete.")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))


# ==========================
# ACD WAVEFORM PROCESSING (ALL CHANNELS, USE waveform_noz)
# ==========================

def process_ae(raw_ae_path, processed_dir, metadata_dir):

    print("Processing ACD waveform dataset (ALL channels, waveform_noz)...")

    train_pkl = raw_ae_path / "acd_model_data_5050_sm_df.pkl"
    test_pkl = raw_ae_path / "acd_test_data_df.pkl"

    if not train_pkl.exists() or not test_pkl.exists():
        raise ValueError("ACD .pkl files not found in raw/ae directory.")

    train_df = pd.read_pickle(train_pkl).reset_index(drop=True)
    test_df = pd.read_pickle(test_pkl).reset_index(drop=True)

    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))

    print("\nTrain channel distribution:")
    print(train_df["channel"].value_counts())

    print("\nTest channel distribution:")
    print(test_df["channel"].value_counts())

    # ------------------------------
    # Use waveform_noz (trimmed)
    # ------------------------------
    FIXED_LENGTH = 4096

    def fix_length(signal, target_len=FIXED_LENGTH):
        signal = np.array(signal, dtype=np.float32)
        length = len(signal)

        if length > target_len:
            start = (length - target_len) // 2
            return signal[start:start + target_len]

        elif length < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:length] = signal
            return padded

        return signal


    X_train = np.stack([
        fix_length(w) for w in train_df["waveform_noz"].values
    ])

    y_train = train_df["crack"].values.astype(np.int64)

    X_test = np.stack([
        fix_length(w) for w in test_df["waveform_noz"].values
    ])

    y_test = test_df["crack"].values.astype(np.int64)

    ensure_dir(processed_dir)

    np.save(Path(processed_dir) / "waveforms_train.npy", X_train)
    np.save(Path(processed_dir) / "labels_train.npy", y_train)
    np.save(Path(processed_dir) / "waveforms_test.npy", X_test)
    np.save(Path(processed_dir) / "labels_test.npy", y_test)

    ensure_dir(metadata_dir)

    pd.DataFrame({
        "label": y_train
    }).to_csv(Path(metadata_dir) / "ae_train_metadata.csv", index=False)

    pd.DataFrame({
        "label": y_test
    }).to_csv(Path(metadata_dir) / "ae_test_metadata.csv", index=False)

    print("\nAE waveform preprocessing complete.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


# ==========================
# MAIN
# ==========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    raw_images = data_root / "raw/images/SDNET2018"
    raw_ae = data_root / "raw/ae"

    processed_images = data_root / "processed/images"
    processed_ae = data_root / "processed/ae"

    metadata_dir = data_root / "metadata"

    if raw_images.exists():
        process_images(raw_images, processed_images, metadata_dir)

    if raw_ae.exists():
        process_ae(raw_ae, processed_ae, metadata_dir)


if __name__ == "__main__":
    main()
