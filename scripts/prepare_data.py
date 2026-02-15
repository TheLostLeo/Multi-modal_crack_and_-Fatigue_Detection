import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]
WINDOW_SIZE = 100
WINDOW_STRIDE = 50


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


# ==========================
# AE PROCESSING
# ==========================

def assign_damage_stage(stress_norm):

    stages = np.zeros_like(stress_norm)

    stages[stress_norm >= 0.25] = 1
    stages[stress_norm >= 0.5] = 2
    stages[stress_norm >= 0.8] = 3

    return stages


def process_ae(raw_ae_path, processed_dir, metadata_dir):

    print("Processing AE dataset...")

    ae_excel = None

    for root, _, files in os.walk(raw_ae_path):
        for file in files:
            if file.endswith(".xlsx") and "Acoustic" in file:
                ae_excel = os.path.join(root, file)
                break
        if ae_excel:
            break

    if ae_excel is None:
        raise ValueError("Acoustic emission Excel file not found.")

    df = pd.read_excel(ae_excel)
    df = df.iloc[2:].reset_index(drop=True)

    all_sequences = []
    all_labels = []
    all_groups = []

    experiment_id = 0

    blocks = [
        (0, 2),
        (14, 16)
    ]

    for time_idx, stress_idx in blocks:

        time = pd.to_numeric(df.iloc[:, time_idx], errors="coerce")
        stress = pd.to_numeric(df.iloc[:, stress_idx], errors="coerce")

        clean_df = pd.DataFrame({
            "time": time,
            "stress": stress
        }).dropna()

        if len(clean_df) < WINDOW_SIZE:
            continue

        time_vals = clean_df["time"].values.astype(float)
        stress_vals = clean_df["stress"].values.astype(float)

        stress_norm = stress_vals / np.max(stress_vals)

        stress_rate = np.gradient(stress_vals, time_vals)
        stress_acc = np.gradient(stress_rate, time_vals)

        features = np.stack([
            stress_norm,
            stress_rate,
            stress_acc
        ], axis=1)

        stages = assign_damage_stage(stress_norm)

        for start in range(0, len(features) - WINDOW_SIZE, WINDOW_STRIDE):
            end = start + WINDOW_SIZE

            all_sequences.append(features[start:end])
            all_labels.append(stages[end - 1])
            all_groups.append(experiment_id)

        experiment_id += 1

    sequences = np.array(all_sequences)
    labels = np.array(all_labels)
    groups = np.array(all_groups)

    ensure_dir(processed_dir)
    np.save(Path(processed_dir) / "ae_sequences.npy", sequences)
    np.save(Path(processed_dir) / "ae_labels.npy", labels)
    np.save(Path(processed_dir) / "ae_groups.npy", groups)

    ensure_dir(metadata_dir)
    pd.DataFrame({
        "sequence_id": np.arange(len(labels)),
        "damage_stage": labels,
        "group_id": groups
    }).to_csv(Path(metadata_dir) / "ae_labels.csv", index=False)

    print("AE processing complete.")
    print("Total sequences:", len(sequences))
    print("Unique groups:", np.unique(groups))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    raw_images = data_root / "raw/images/SDNET2018"
    raw_ae = data_root / "raw/ae/metro_tunnel"

    processed_images = data_root / "processed/images"
    processed_ae = data_root / "processed/ae"

    metadata_dir = data_root / "metadata"

    process_images(raw_images, processed_images, metadata_dir)
    process_ae(raw_ae, processed_ae, metadata_dir)


if __name__ == "__main__":
    main()

