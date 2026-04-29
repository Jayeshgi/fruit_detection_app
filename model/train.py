"""
train.py
──────────────────────────────────────────────────────────────────────────────
Main training script for the Fruit Detection Model.

Usage:
    python train.py

What it does:
    1. Loads the Fruits-360 dataset (Training & Test splits)
    2. Creates a MobileNetV3 model with transfer learning
    3. Trains the model for NUM_EPOCHS
    4. Evaluates on the test set after each epoch
    5. Saves the best model weights and class names to saved_models/
──────────────────────────────────────────────────────────────────────────────
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_SAVE_PATH, CLASS_NAMES_FILE,
)
from dataset import get_dataloaders
from fruit_model import create_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch using Mixed Precision (AMP) for speed."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize GradScaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Runs the forward pass with autocasting
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        acc = 100 * correct / total
        progress_bar.set_postfix({"loss": f"{loss.item():.2f}", "acc": f"{acc:.1f}%"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on the test set and return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Testing ", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    print("=" * 60)
    print("  Fruit Detection Model - Training Pipeline")
    print("=" * 60)

    # ── Device Setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    train_loader, test_loader, class_names, num_classes = get_dataloaders()

    # ── Create Model ──────────────────────────────────────────────────────
    print("\n[2/4] Creating model...")
    model = create_model(num_classes=num_classes)
    model = model.to(device)

    # ── Loss & Optimizer ──────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Learning rate scheduler — reduce LR when test accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    # ── Training Loop ─────────────────────────────────────────────────────
    print(f"\n[3/4] Training for {NUM_EPOCHS} epochs...\n")

    best_test_acc = 0.0
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device,
        )

        # Step scheduler
        scheduler.step(test_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n"
            f"  Time: {epoch_time:.1f}s"
        )

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = os.path.join(MODEL_SAVE_PATH, "best_fruit_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "num_classes": num_classes,
                "class_names": class_names,
            }, save_path)
            print(f"  * New best model saved! (Test Acc: {test_acc:.2f}%)\n")
        else:
            print()

    # ── Save Class Names ──────────────────────────────────────────────────
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"  Model saved to:    {os.path.join(MODEL_SAVE_PATH, 'best_fruit_model.pth')}")
    print(f"  Class names saved: {CLASS_NAMES_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
