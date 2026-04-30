"""
dataset.py
──────────────────────────────────────────────────────────────────────────────
Provides PyTorch Dataset / DataLoader utilities for the Fruits-360 dataset.

The Fruits-360 dataset is already organised as:
    Training/
        Apple Braeburn/
            img1.jpg
            img2.jpg
            ...
        Banana/
            ...
    Test/
        Apple Braeburn/
            ...

torchvision.datasets.ImageFolder handles this structure automatically.
──────────────────────────────────────────────────────────────────────────────
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD,
)


def get_train_transforms():
    """
    Training transforms include data augmentation to make the model
    generalize better to real-world images (not just clean white-bg ones).
    """
    return transforms.Compose([
    # 1. Randomly zoom and crop (harder: 60-100% scale forces partial views)
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    
    # 2. Random flips and rotations
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    
    # 3. Perspective distortion (simulates photos taken at angles)
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    
    # 4. Randomly change brightness, contrast, and color
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    
    # 5. Add a bit of blur (simulates a shaky hand or low-res camera)
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    
    # 6. Occasionally turn the image black and white (forces learning shapes)
    transforms.RandomGrayscale(p=0.15),
    
    # 7. Final conversion and normalization
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])



def get_test_transforms():
    """
    Test/inference transforms — NO augmentation, only resize and normalize.
    This MUST match what we do at inference time in the API.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders():
    """
    Returns:
        train_loader: DataLoader for training set
        test_loader:  DataLoader for test/validation set
        class_names:  List[str] of class names (folder names)
        num_classes:  int, number of fruit classes
    """
    # Verify dataset exists
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found at: {TRAIN_DIR}\n"
            f"Run 'python download_dataset.py' first to download the dataset."
        )
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory not found at: {TEST_DIR}\n"
            f"Run 'python download_dataset.py' first to download the dataset."
        )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )
    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_test_transforms(),
    )

    # Get class info
    class_names = train_dataset.classes       # list of folder names
    num_classes = len(class_names)

    print(f"[i] Found {num_classes} fruit classes")
    print(f"    Training images : {len(train_dataset)}")
    print(f"    Test images     : {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, class_names, num_classes
