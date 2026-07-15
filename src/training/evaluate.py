import torch
from dataset import TinyDrowsyDataset
from augmentations import get_validation_transform
from model import TinyEyeNetV2
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(checkpoint_path, data_split='test'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    class_names = checkpoint['class_names']
    model = TinyEyeNetV2(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = TinyDrowsyDataset(Path('dataset/processed')/data_split, transform=get_validation_transform())
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.numpy())
            labels.extend(lbls.numpy())

    print(classification_report(labels, preds, target_names=class_names))
    return confusion_matrix(labels, preds)

if __name__ == '__main__':
    evaluate('training_output/best_model.pth')