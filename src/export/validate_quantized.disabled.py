"""
validate_quantized.py
---------------------
Compare FP32 ONNX vs INT8 ESP-DL model accuracy on test dataset.

This uses the official Espressif workflow:
1. Quantize model with espdl_quantize_onnx()
2. Use returned quantized graph to run inference on PC via TorchExecutor
3. Compare against FP32 ONNX model

Based on ESP-DL tutorial: https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import TinyDrowsyDataset
from data.augmentations import get_validation_transform

import onnxruntime as ort
import onnx
from onnxsim import simplify

# ==========================================================
# CORRECT ESP-PPQ IMPORTS
# ==========================================================

# ESP-PPQ uses ppq.api for model loading and execution
from ppq.api import load_onnx_graph
from esp_ppq import TorchExecutor
from ppq import QuantizationSetting
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
ONNX_PATH = PROJECT_ROOT / "training_output" / "tinydrowsy_fp32.onnx"
ESPDL_PATH = PROJECT_ROOT / "training_output" / "tinydrowsy_int8.espdl"
CALIB_DIR = PROJECT_ROOT / "dataset" / "calibration"
TEST_DIR = PROJECT_ROOT / "dataset" / "processed" / "test"

BATCH_SIZE = 64
IMAGE_SIZE = 64
CALIB_STEPS = 32
DEVICE = "cpu"
TARGET = "esp32s3"
NUM_OF_BITS = 8
INPUT_SHAPE = [1, IMAGE_SIZE, IMAGE_SIZE]  # [channels, height, width]

# ==========================================================
# CALIBRATION DATASET (Required for quantization)
# ==========================================================

class CalibrationDataset(torch.utils.data.Dataset):
    """Load calibration images from folder."""
    
    def __init__(self, path, img_size=64):
        self.path = Path(path)
        self.img_size = img_size
        self.image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            self.image_paths.extend(self.path.glob(f"*{ext}"))
        
        print(f"  Found {len(self.image_paths)} calibration images")
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        return self.transform(img)


def prepare_calibration_dataset():
    """Create calibration dataset folder if it doesn't exist."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    
    if not any(CALIB_DIR.iterdir()):
        print("📂 Calibration dataset empty. Copying samples from training set...")
        train_dir = PROJECT_ROOT / "dataset" / "processed" / "train"
        import shutil
        copied = 0
        for class_name in ["open", "closed"]:
            class_dir = train_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))
                for img_path in images[:50]:
                    dest = CALIB_DIR / f"{class_name}_{img_path.name}"
                    shutil.copy(img_path, dest)
                    copied += 1
        print(f"  Copied {copied} images to {CALIB_DIR}")
    
    return CALIB_DIR


# ==========================================================
# INFERENCE HELPERS
# ==========================================================

def load_fp32_model(onnx_path):
    """Load FP32 ONNX model."""
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def get_quantized_executor(onnx_path, calib_dataloader, collate_fn):
    """
    Quantize the model and return a TorchExecutor for PC inference.
    
    This is the CORRECT official workflow from ESP-DL documentation:
    - espdl_quantize_onnx() returns a quantized graph (BaseGraph)
    - Build TorchExecutor from that graph for evaluation
    """
    
    print("⚙️ Quantizing model for validation...")
    
    # Load and simplify ONNX
    model = onnx.load(str(onnx_path))
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, str(onnx_path))
    
    # Configure quantization
    quant_setting = QuantizationSettingFactory.espdl_setting()
    
    # Use percentile calibration for better results
    quant_setting.quantize_activation_setting.calib_algorithm = "percentile"
    
    # Enable bias correction
    quant_setting.bias_correct = True
    quant_setting.bias_correct_setting.block_size = 4
    quant_setting.bias_correct_setting.steps = 32
    
    # Enable TQT optimization
    quant_setting.tqt_optimization = True
    quant_setting.tqt_optimization_setting.lr = 1e-5
    quant_setting.tqt_optimization_setting.steps = 500
    quant_setting.tqt_optimization_setting.block_size = 4
    quant_setting.tqt_optimization_setting.is_scale_trainable = True
    quant_setting.tqt_optimization_setting.int_lambda = 0.25
    quant_setting.tqt_optimization_setting.collecting_device = DEVICE
    
    # Run quantization - this returns a quantized graph
    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(ESPDL_PATH),  # Export for deployment
        calib_dataloader=calib_dataloader,
        calib_steps=CALIB_STEPS,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=0,
        inputs=None,
    )
    
    print("✅ Quantization complete. Building executor...")
    
    # Build TorchExecutor from quantized graph for PC evaluation
    executor = TorchExecutor(graph=quant_graph, device=DEVICE)
    
    return executor


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_fp32(session, input_name, output_name, dataloader):
    """Evaluate FP32 ONNX model."""
    all_preds = []
    all_labels = []
    
    print("  Evaluating FP32 model...")
    for images, labels in tqdm(dataloader, desc="FP32 Inference"):
        inputs = {input_name: images.numpy().astype(np.float32)}
        outputs = session.run([output_name], inputs)
        preds = np.argmax(outputs[0], axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def evaluate_int8(executor, dataloader):
    """Evaluate INT8 quantized model using TorchExecutor."""
    all_preds = []
    all_labels = []
    
    print("  Evaluating INT8 model...")
    for images, labels in tqdm(dataloader, desc="INT8 Inference"):
        # Run inference on quantized graph
        outputs = executor(images)
        preds = np.argmax(outputs, axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("  TinyDrowsy Quantization Validation")
    print("  FP32 vs INT8 Accuracy Comparison")
    print("  Official ESP-PPQ Workflow")
    print("=" * 60)
    
    # 1. Load test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = TinyDrowsyDataset(
        TEST_DIR,
        transform=get_validation_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = test_dataset.class_names()
    print(f"  Classes: {class_names}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # 2. Prepare calibration dataset
    print("\n📂 Preparing calibration dataset...")
    calib_dir = prepare_calibration_dataset()
    calib_dataset = CalibrationDataset(calib_dir, img_size=IMAGE_SIZE)
    
    def collate_fn(batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(DEVICE)

        return torch.stack(batch).to(DEVICE)
    
    calib_loader = DataLoader(
        dataset=calib_dataset,
        batch_size=32,
        shuffle=False,
    )
    
    # 3. Load FP32 model
    print("\n🔧 Loading FP32 ONNX model...")
    session, input_name, output_name = load_fp32_model(ONNX_PATH)
    
    # 4. Get quantized executor
    print("\n🔧 Quantizing and creating INT8 executor...")
    int8_executor = get_quantized_executor(ONNX_PATH, calib_loader, collate_fn)
    
    # 5. Evaluate FP32
    fp32_preds, true_labels = evaluate_fp32(session, input_name, output_name, test_loader)
    fp32_acc = np.mean(fp32_preds == true_labels)
    
    # 6. Evaluate INT8
    int8_preds, _ = evaluate_int8(int8_executor, test_loader)
    int8_acc = np.mean(int8_preds == true_labels)
    
    # 7. Compare results
    print("\n" + "=" * 60)
    print("  Results Comparison")
    print("=" * 60)
    print(f"FP32 Accuracy:  {fp32_acc:.4%}")
    print(f"INT8 Accuracy:  {int8_acc:.4%}")
    print(f"Accuracy Drop:  {fp32_acc - int8_acc:.4%} ({int8_acc/fp32_acc:.2%} of original)")
    
    # 8. Classification reports
    print("\n📊 FP32 Classification Report:")
    print(classification_report(true_labels, fp32_preds, target_names=class_names))
    
    print("\n📊 INT8 Classification Report:")
    print(classification_report(true_labels, int8_preds, target_names=class_names))
    
    # 9. Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    fp32_cm = confusion_matrix(true_labels, fp32_preds)
    int8_cm = confusion_matrix(true_labels, int8_preds)
    
    ConfusionMatrixDisplay(fp32_cm, display_labels=class_names).plot(ax=ax1)
    ax1.set_title(f'FP32 - {fp32_acc:.2%}')
    
    ConfusionMatrixDisplay(int8_cm, display_labels=class_names).plot(ax=ax2)
    ax2.set_title(f'INT8 - {int8_acc:.2%}')
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "training_output" / "quantization_validation.png", dpi=150)
    print("\n📊 Quantization validation plot saved.")
    
    # 10. Summary report
    summary = {
        "fp32_accuracy": float(fp32_acc),
        "int8_accuracy": float(int8_acc),
        "accuracy_drop": float(fp32_acc - int8_acc),
        "test_samples": len(test_dataset)
    }
    
    with open(PROJECT_ROOT / "training_output" / "quantization_validation.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n✅ Validation complete!")
    print(f"📁 Results saved to: training_output/quantization_validation.json")
    
    return fp32_acc, int8_acc


if __name__ == "__main__":
    main()