import hashlib
import csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_ROOT = Path("dataset/raw/train(MRL+CEW)")

CLASS_FOLDERS = {
    "open": DATASET_ROOT / "Open_Eyes",
    "closed": DATASET_ROOT / "Closed_Eyes",
}

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

CSV_REPORT = REPORT_DIR / "dataset_audit.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

def calculate_brightness(gray_image):
    return float(np.mean(gray_image))


def calculate_blur(gray_image):
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def md5_hash(filepath):
    hash_md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()

def analyze_image(filepath):

    try:

        image = cv2.imread(str(filepath))

        if image is None:
            return {
                "valid": False
            }

        height, width = image.shape[:2]

        channels = 1 if len(image.shape) == 2 else image.shape[2]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = calculate_brightness(gray)

        blur = calculate_blur(gray)

        return {

            "valid": True,

            "width": width,

            "height": height,

            "channels": channels,

            "brightness": brightness,

            "blur": blur,

            "hash": md5_hash(filepath)

        }

    except Exception:

        return {
            "valid": False
        }

def collect_images():

    rows = []

    duplicate_counter = Counter()

    print("\nScanning images...\n")

    for class_name, folder in CLASS_FOLDERS.items():

        image_files = []

        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(folder.rglob(f"*{ext}"))

        for image_path in tqdm(image_files, desc=class_name):

            result = analyze_image(image_path)

            if result["valid"]:

                duplicate_counter[result["hash"]] += 1

            rows.append({

                "file": image_path.name,

                "path": str(image_path),

                "class": class_name,

                **result

            })

    return rows, duplicate_counter

# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(df):

    print("\n" + "=" * 60)
    print(" TinyDrowsy Dataset Audit ")
    print("=" * 60)

    total_images = len(df)

    valid_images = df["valid"].sum()

    invalid_images = total_images - valid_images

    print(f"Total Images      : {total_images}")
    print(f"Readable Images   : {valid_images}")
    print(f"Unreadable Images : {invalid_images}")

    print("\nClass Distribution")

    print(df["class"].value_counts())

    print("\nChannels")

    print(df["channels"].value_counts())

    print("\nMost Common Image Sizes")

    size_counts = (
        df[df["valid"]]
        .groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
    )

    print(size_counts.head(10))

    print("\nAverage Brightness")

    print(
        df.groupby("class")["brightness"]
        .mean()
        .round(2)
    )

    print("\nAverage Blur")

    print(
        df.groupby("class")["blur"]
        .mean()
        .round(2)
    )

def report_duplicates(df):

    duplicates = df[
        df.duplicated("hash", keep=False)
    ]

    if len(duplicates) == 0:

        print("\nDuplicate Images : 0")

        return

    print(f"\nDuplicate Images : {len(duplicates)}")

    duplicate_csv = REPORT_DIR / "duplicate_images.csv"

    duplicates.to_csv(
        duplicate_csv,
        index=False,
    )

    print(f"Duplicate report saved to:")
    print(duplicate_csv)

def save_report(df):

    df.to_csv(
        CSV_REPORT,
        index=False,
    )

    print("\nCSV Report Saved")

    print(CSV_REPORT)

# ==========================================================
# MAIN
# ==========================================================

def main():

    rows, duplicate_counter = collect_images()

    df = pd.DataFrame(rows)

    save_report(df)

    print_summary(df)

    report_duplicates(df)

    print("\nAudit Completed Successfully.")


if __name__ == "__main__":

    main()