import os
import random
import shutil
import csv
from pathlib import Path

random.seed(42)

NEG_IMG_SRC = r"E:\OPERA\Opus_5 - Crack & Fatigue Detection\image_concrete\Negative"
POS_IMG_SRC = r"E:\OPERA\Opus_5 - Crack & Fatigue Detection\image_concrete\Positive"

AUDIO_ROOT = r"E:\OPERA\Opus_5 - Crack & Fatigue Detection\audio_slider\slider"

DATASET_ROOT = r"E:\OPERA\Opus_5 - Crack & Fatigue Detection\Multimodal Image & Acoustic Dataset"

IMG_NEG = os.path.join(DATASET_ROOT, "image", "negative")
IMG_POS = os.path.join(DATASET_ROOT, "image", "positive")

AUD_NEG = os.path.join(DATASET_ROOT, "audio", "negative")
AUD_POS = os.path.join(DATASET_ROOT, "audio", "positive")

for p in [IMG_NEG, IMG_POS, AUD_NEG, AUD_POS]:
    os.makedirs(p, exist_ok=True)

normal_audio = []
abnormal_audio = []

machine_ids = ["id_00", "id_02", "id_04", "id_06"]

for mid in machine_ids:
    normal_path = os.path.join(AUDIO_ROOT, mid, "normal")
    abnormal_path = os.path.join(AUDIO_ROOT, mid, "abnormal")
    for f in os.listdir(normal_path):
        if f.endswith(".wav"):
            normal_audio.append(os.path.join(normal_path, f))
    for f in os.listdir(abnormal_path):
        if f.endswith(".wav"):
            abnormal_audio.append(os.path.join(abnormal_path, f))

print("Normal audio:", len(normal_audio))
print("Abnormal audio:", len(abnormal_audio))

negative_images = [
    os.path.join(NEG_IMG_SRC, f)
    for f in os.listdir(NEG_IMG_SRC)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

positive_images = [
    os.path.join(POS_IMG_SRC, f)
    for f in os.listdir(POS_IMG_SRC)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

normal_count = len(normal_audio)
abnormal_count = len(abnormal_audio)

selected_negative_imgs = random.sample(negative_images, normal_count)
selected_positive_imgs = random.sample(positive_images, abnormal_count)

random.shuffle(normal_audio)
random.shuffle(abnormal_audio)
random.shuffle(selected_negative_imgs)
random.shuffle(selected_positive_imgs)

csv_rows = []
for i, (img, aud) in enumerate(zip(selected_negative_imgs, normal_audio)):
    img_name = f"neg_img_{i}.jpg"
    aud_name = f"neg_audio_{i}.wav"
    img_dest = os.path.join(IMG_NEG, img_name)
    aud_dest = os.path.join(AUD_NEG, aud_name)
    shutil.copy(img, img_dest)
    shutil.copy(aud, aud_dest)
    csv_rows.append([img_dest, aud_dest, "normal"])

for i, (img, aud) in enumerate(zip(selected_positive_imgs, abnormal_audio)):
    img_name = f"pos_img_{i}.jpg"
    aud_name = f"pos_audio_{i}.wav"
    img_dest = os.path.join(IMG_POS, img_name)
    aud_dest = os.path.join(AUD_POS, aud_name)
    shutil.copy(img, img_dest)
    shutil.copy(aud, aud_dest)
    csv_rows.append([img_dest, aud_dest, "abnormal"])
random.shuffle(csv_rows)

csv_path = os.path.join(DATASET_ROOT, "dataset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "audio_path", "label"])
    writer.writerows(csv_rows)

print("\nDataset creation complete.")
print("Total samples:", len(csv_rows))
print("Dataset location:", DATASET_ROOT)