"""
quantize_espdl.py
-----------------
Quantize ONNX model to INT8 ESP-DL (.espdl) format using ESP-PPQ.

Official Espressif workflow:
1. Load ONNX model
2. Prepare calibration dataset
3. Configure quantization settings
4. Quantize and export .espdl
"""

import os
import sys
from pathlib import Path

import onnx
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from onnxsim import simplify

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==========================================================
# CONFIGURATION
# ==========================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ONNX_PATH = PROJECT_ROOT / "training_output" / "tinydrowsy_fp32.onnx"
ESPDL_PATH = PROJECT_ROOT / "training_output" / "tinydrowsy_int8.espdl"
CALIB_DIR = PROJECT_ROOT / "dataset" / "calibration"

# Model parameters
IMAGE_SIZE = 64
INPUT_SHAPE = [1, IMAGE_SIZE, IMAGE_SIZE]  # [channels, height, width]
BATCH_SIZE = 32
CALIB_STEPS = 32

# Quantization parameters
TARGET = "esp32s3"  # or "esp32p4", "esp32"
NUM_OF_BITS = 8

# Device
DEVICE = "cpu"


# ==========================================================
# CALIBRATION DATASET
# ==========================================================

class CalibrationDataset(Dataset):
    """Load calibration images from folder."""
    
    def __init__(self, path, img_size=64):
        self.path = Path(path)
        self.img_size = img_size
        
        # Collect all images
        self.image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            self.image_paths.extend(self.path.glob(f"*{ext}"))
        
        print(f"  Found {len(self.image_paths)} calibration images")
        
        # Transform: same as training preprocessing
        # Normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Grayscale
        return self.transform(img)


# ==========================================================
# QUANTIZATION
# ==========================================================

def prepare_calibration_dataset():
    """Create calibration dataset folder if it doesn't exist."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    
    # If empty, copy 100 images from training set
    if not any(CALIB_DIR.iterdir()):
        print("📂 Calibration dataset empty. Copying samples from training set...")
        train_dir = PROJECT_ROOT / "dataset" / "processed" / "train"
        
        # Get samples from each class
        copied = 0
        for class_name in ["open", "closed"]:
            class_dir = train_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))
                # Take first 50 images from each class
                for img_path in images[:50]:
                    dest = CALIB_DIR / f"{class_name}_{img_path.name}"
                    import shutil
                    shutil.copy(img_path, dest)
                    copied += 1
        
        print(f"  Copied {copied} images to {CALIB_DIR}")
    
    return CALIB_DIR


def quantize_model():
    """Quantize ONNX model and export .espdl."""
    
    print("=" * 60)
    print("  TinyDrowsy Quantization")
    print("  ESP-PPQ INT8 Quantization")
    print("=" * 60)
    
    # 1. Check ONNX model exists
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")
    print(f"✅ ONNX model: {ONNX_PATH}")
    
    # 2. Prepare calibration dataset
    print("\n📂 Preparing calibration dataset...")
    calib_dir = prepare_calibration_dataset()
    
    # 3. Load and simplify ONNX model
    print("\n🔧 Loading ONNX model...")
    model = onnx.load(str(ONNX_PATH))
    
    # Simplify ONNX graph (optional but recommended)
    print("  Simplifying ONNX graph...")
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, str(ONNX_PATH))
    
    # 4. Create calibration dataset
    print(f"\n📊 Creating calibration dataset from {calib_dir}...")
    calib_dataset = CalibrationDataset(calib_dir, img_size=IMAGE_SIZE)
    dataloader = DataLoader(
        dataset=calib_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    def collate_fn(batch):
        """Collate function for DataLoader."""
        return batch.to(DEVICE)
    
    # 5. Configure quantization settings
    print(f"\n⚙️ Configuring quantization (INT8, {TARGET})...")
    
    quant_setting = QuantizationSettingFactory.espdl_setting()
    
    # Use percentile calibration for better results
    quant_setting.quantize_activation_setting.calib_algorithm = "percentile"
    
    # Enable bias correction
    quant_setting.bias_correct = True
    quant_setting.bias_correct_setting.block_size = 4
    quant_setting.bias_correct_setting.steps = 32
    
    # Enable TQT optimization for better accuracy
    quant_setting.tqt_optimization = True
    tqt_setting = quant_setting.tqt_optimization_setting
    tqt_setting.lr = 1e-5
    tqt_setting.steps = 500
    tqt_setting.block_size = 4
    tqt_setting.is_scale_trainable = True
    tqt_setting.int_lambda = 0.25
    tqt_setting.collecting_device = DEVICE
    
    # 6. Quantize and export
    print(f"\n🚀 Starting quantization...")
    print(f"  Target: {TARGET}")
    print(f"  Bits: {NUM_OF_BITS}")
    print(f"  Input shape: {INPUT_SHAPE}")
    print(f"  Calibration steps: {CALIB_STEPS}")
    
    try:
        quant_graph = espdl_quantize_onnx(
            onnx_import_file=str(ONNX_PATH),
            espdl_export_file=str(ESPDL_PATH),
            calib_dataloader=dataloader,
            calib_steps=CALIB_STEPS,
            input_shape=[1] + INPUT_SHAPE,  # [batch, channels, height, width]
            target=TARGET,
            num_of_bits=NUM_OF_BITS,
            collate_fn=collate_fn,
            setting=quant_setting,
            device=DEVICE,
            error_report=True,
            skip_export=False,
            export_test_values=True,
            verbose=1,
            inputs=None,
        )
        
        print("\n" + "=" * 60)
        print("✅ Quantization completed successfully!")
        print(f"📁 ESPDL model: {ESPDL_PATH}")
        print(f"   Size: {ESPDL_PATH.stat().st_size / 1024:.2f} KB")
        print("=" * 60)
        
        return quant_graph
        
    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")
        raise


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    quantize_model()