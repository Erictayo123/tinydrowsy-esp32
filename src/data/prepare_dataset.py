"""
prepare_dataset.py
------------------
Multi-source dataset preparation with source-aware splitting.

Strategy:
    1. MRL+CEW: Primary training source (large, diverse)
    2. DDD: Secondary training source (driver-specific)
    3. Selfies: Validation/Test source (real-world generalization)

This ensures the model generalizes beyond the training distribution.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import json
import random
import shutil
import numpy as np
from collections import defaultdict

# ==========================================================
# CONFIGURATION
# ==========================================================

SOURCE_ROOTS = [
    Path("dataset/raw/train(MRL+CEW)"),           # Primary training
    Path("dataset/raw/Driver Drowsiness Dataset (DDD)"),  # Secondary training
    Path("dataset/raw/selfies"),                  # Real-world validation/test
]

CLASS_NAMES = ["open", "closed"]

# Source-specific folder mappings
SOURCE_FOLDER_MAP = {
    "train(MRL+CEW)": {
        "open": ["Open_Eyes"],
        "closed": ["Closed_Eyes"]
    },
    "selfies": {
        "open": ["Open_Eyes"],
        "closed": ["Closed_Eyes"]
    },
    "Driver Drowsiness Dataset (DDD)": {
        "open": ["Non Drowsy"],
        "closed": ["Drowsy"]
    }
}

OUTPUT_ROOT = Path("dataset/processed")
TRAIN_FOLDER = OUTPUT_ROOT / "train"
VAL_FOLDER = OUTPUT_ROOT / "validation"
TEST_FOLDER = OUTPUT_ROOT / "test"

IMAGE_SIZE = 64
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
JPEG_QUALITY = 95

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================================
# UTILITIES
# ==========================================================

def create_directory_structure():
    """Create train/val/test directories with class subfolders."""
    folders = []
    for split in ["train", "validation", "test"]:
        for cls in CLASS_NAMES:
            folders.append(OUTPUT_ROOT / split / cls)
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure in {OUTPUT_ROOT}")

def clear_processed_dataset():
    """Remove existing processed dataset."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    create_directory_structure()

def list_images_from_source(source_root, class_name):
    """List images from a source using source-specific folder mappings."""
    images = []
    source_name = source_root.name
    alias_list = SOURCE_FOLDER_MAP.get(source_name, {}).get(class_name, [])
    
    for alias in alias_list:
        class_folder = source_root / alias
        if class_folder.exists() and class_folder.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                images.extend(class_folder.glob(f"*{ext}"))
    
    return sorted(images)

def preprocess_image(image):
    """Convert to grayscale, resize, and optionally equalize."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return gray

def save_image(image, output_path):
    """Save image as JPEG with quality setting."""
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def collect_all_images():
    """Collect images from all sources."""
    print("\n📂 Scanning image sources...")
    image_map = {cls: [] for cls in CLASS_NAMES}
    source_stats = defaultdict(lambda: defaultdict(int))
    
    for source_root in SOURCE_ROOTS:
        if not source_root.exists():
            print(f"⚠️  Source not found: {source_root}")
            continue
        
        print(f"\n  Source: {source_root.name}")
        for class_name in CLASS_NAMES:
            images = list_images_from_source(source_root, class_name)
            image_map[class_name].extend(images)
            source_stats[source_root.name][class_name] = len(images)
            print(f"    {class_name}: {len(images)} images")
    
    return image_map, source_stats

def process_images(image_paths, class_name):
    """Process and save images for a given class."""
    train_imgs, temp_imgs = train_test_split(
        image_paths, train_size=TRAIN_RATIO, random_state=RANDOM_SEED
    )
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_ratio_adj, random_state=RANDOM_SEED
    )
    
    splits = [
        ("train", train_imgs, TRAIN_FOLDER),
        ("validation", val_imgs, VAL_FOLDER),
        ("test", test_imgs, TEST_FOLDER),
    ]
    
    stats = {}
    for split_name, img_list, split_folder in splits:
        output_folder = split_folder / class_name
        saved, failed = 0, 0
        
        for img_path in tqdm(img_list, desc=f"{class_name} [{split_name}]"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    failed += 1
                    continue
                
                processed = preprocess_image(img)
                filename = f"{img_path.stem}.jpg"
                save_image(processed, output_folder / filename)
                saved += 1
                
            except Exception as e:
                failed += 1
                print(f"\n❌ Error processing {img_path}: {e}")
        
        stats[split_name] = {"saved": saved, "failed": failed}
    
    return stats

def build_metadata(source_stats):
    """Create comprehensive dataset metadata."""
    def count_images(folder):
        return sum(1 for _ in folder.rglob("*.jpg"))
    
    metadata = {
        "image_size": IMAGE_SIZE,
        "channels": 1,
        "format": "grayscale",
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO
        },
        "classes": CLASS_NAMES,
        "source_statistics": dict(source_stats),
        "split_counts": {
            "train": {
                "open": count_images(TRAIN_FOLDER / "open"),
                "closed": count_images(TRAIN_FOLDER / "closed")
            },
            "validation": {
                "open": count_images(VAL_FOLDER / "open"),
                "closed": count_images(VAL_FOLDER / "closed")
            },
            "test": {
                "open": count_images(TEST_FOLDER / "open"),
                "closed": count_images(TEST_FOLDER / "closed")
            }
        }
    }
    return metadata

def save_metadata(metadata):
    """Save metadata to JSON."""
    with open(OUTPUT_ROOT / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n📄 Metadata saved to {OUTPUT_ROOT}/dataset_info.json")

def print_summary(metadata):
    """Print a formatted summary."""
    print("\n" + "=" * 60)
    print("  TinyDrowsy Dataset Preparation")
    print("=" * 60)
    print(f"Image Size: {metadata['image_size']}×{metadata['image_size']}")
    print(f"Classes: {metadata['classes']}")
    
    print("\n📊 Source Statistics:")
    for source, counts in metadata['source_statistics'].items():
        print(f"  {source}: {counts}")
    
    print("\n📊 Split Counts:")
    for split, counts in metadata['split_counts'].items():
        total = counts['open'] + counts['closed']
        print(f"  {split}: {total} images ({counts['open']} open, {counts['closed']} closed)")
    
    total = sum(sum(c.values()) for c in metadata['split_counts'].values())
    print(f"\n✅ Total Images: {total}")

# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("  TinyDrowsy Dataset Preparation")
    print("  Multi-Source Merge with Selfies Validation")
    print("=" * 60)
    
    # Clear and create structure
    clear_processed_dataset()
    
    # Collect images
    image_map, source_stats = collect_all_images()
    
    # Verify all classes have images
    for cls, paths in image_map.items():
        if len(paths) == 0:
            raise RuntimeError(f"❌ No images found for class: {cls}")
    
    # Process each class
    print("\n🔄 Processing images...")
    all_stats = {}
    for class_name, paths in image_map.items():
        stats = process_images(paths, class_name)
        all_stats[class_name] = stats
    
    # Build metadata
    metadata = build_metadata(source_stats)
    metadata["processing_stats"] = all_stats
    save_metadata(metadata)
    print_summary(metadata)
    
    print("\n✅ Dataset preparation completed!")
    print(f"📁 Processed dataset: {OUTPUT_ROOT}")
    print("🚀 Ready for training!")

if __name__ == "__main__":
    main()