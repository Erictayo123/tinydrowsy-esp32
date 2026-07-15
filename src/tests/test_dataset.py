from dataset import TinyDrowsyDataset
from augmentations import get_train_transform


dataset = TinyDrowsyDataset(

    "dataset/processed/train",

    transform=get_train_transform()

)

print()

print("Dataset Loaded Successfully")

print()

print("Images :", len(dataset))

print()

print("Classes :", dataset.class_names())

print()

print("Distribution")

print(dataset.class_distribution())

image, label = dataset[0]

print()

print(image.shape)

print(label)