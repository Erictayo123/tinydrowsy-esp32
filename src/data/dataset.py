"""
dataset.py
----------
TinyDrowsy Dataset Loader with automatic class discovery.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TinyDrowsyDataset(Dataset):
    """
    Dataset loader for TinyDrowsy eye-state classification.
    
    Automatically discovers class folders (e.g., open, closed) from root_dir.
    Supports grayscale conversion and custom transforms.
    
    Args:
        root_dir: Path to dataset root (e.g., dataset/processed/train)
        transform: Torchvision transforms to apply
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Auto-discover class folders
        self.class_names_list = sorted([
            folder.name for folder in self.root_dir.iterdir()
            if folder.is_dir()
        ])
        self.CLASS_TO_INDEX = {
            name: idx for idx, name in enumerate(self.class_names_list)
        }
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths with their class labels."""
        for class_name in self.class_names_list:
            class_folder = self.root_dir / class_name
            if not class_folder.exists():
                continue
            
            for image_path in sorted(class_folder.glob("*")):
                # Only include image files
                if image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
                ]:
                    self.samples.append(
                        (image_path, self.CLASS_TO_INDEX[class_name])
                    )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")  # Grayscale
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def num_classes(self):
        """Return number of classes."""
        return len(self.CLASS_TO_INDEX)
    
    def class_names(self):
        """Return list of class names."""
        return self.class_names_list
    
    def class_distribution(self):
        """Return class distribution as dict."""
        counts = {name: 0 for name in self.class_names_list}
        for _, label in self.samples:
            class_name = self.class_names_list[label]
            counts[class_name] += 1
        return counts