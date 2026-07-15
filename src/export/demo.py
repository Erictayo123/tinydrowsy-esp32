import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = Path("training_output/tinydrowsy_fp32.onnx")
CLASS_NAMES = ["closed", "open"]
IMAGE_SIZE = 64

session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    inp = img.astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5   # normalize to [-1,1]
    inp = inp[np.newaxis, np.newaxis, :, :]
    return inp

def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict_image(img_path):
    inp = preprocess_image(img_path)
    logits = session.run([output_name], {input_name: inp})[0]
    probs = softmax(logits)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], probs[pred_idx]

if __name__ == "__main__":
    # Example: place some test images (64x64 grayscale eye crops) in a folder.
    test_folder = Path("test_images")  # create this folder with sample eye crops
    for img_path in test_folder.glob("*.jpg"):
        cls, conf = predict_image(img_path)
        print(f"{img_path.name} -> {cls} (confidence {conf:.4f})")