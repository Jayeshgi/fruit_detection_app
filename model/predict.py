"""
predict.py
------------------------------------------------------------------------------
Inference script - load the trained model and predict the fruit in an image.

Usage:
    python predict.py --image path/to/fruit_image.jpg

This script:
    1. Loads the best saved model from saved_models/
    2. Preprocesses the input image (same transforms as test time)
    3. Runs inference and prints the top-5 predictions with confidence scores
------------------------------------------------------------------------------
"""
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from config import (
    MODEL_SAVE_PATH, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, MODEL_NAME,
)
from fruit_model import create_model


def load_trained_model(model_path: str = None, device: str = "cpu"):
    """
    Load the trained model and class names from a checkpoint file.

    Returns:
        model:       The PyTorch model in eval mode
        class_names: List of class name strings
        device:      torch.device being used
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_PATH, "best_fruit_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at: {model_path}\n"
            f"Run 'python train.py' first to train the model."
        )

    device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = create_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded from: {model_path}")
    print(f"    Trained accuracy : {checkpoint['test_acc']:.2f}%")
    print(f"    Number of classes: {num_classes}")

    return model, class_names, device


def preprocess_image(image_path: str):
    """
    Load and preprocess an image for inference.
    Uses the EXACT SAME transforms as test-time evaluation.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def predict(image_path: str, top_k: int = 5):
    """
    Predict the fruit in the given image.

    Args:
        image_path: Path to the fruit image
        top_k:      Number of top predictions to show

    Returns:
        predictions: List of (class_name, confidence%) tuples
    """
    model, class_names, device = load_trained_model()

    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    # Format results
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = class_names[idx.item()]
        confidence = prob.item() * 100
        predictions.append((class_name, confidence))

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict fruit from an image")
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the fruit image to classify",
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Fruit Detection - Prediction")
    print("=" * 50)

    predictions = predict(args.image, top_k=args.top_k)

    print(f"\n  Image: {args.image}")
    print(f"\n  {'Rank':<6} {'Fruit':<30} {'Confidence':>10}")
    print("  " + "-" * 48)
    for rank, (name, conf) in enumerate(predictions, 1):
        bar = "#" * int(conf / 2.5)
        print(f"  {rank:<6} {name:<30} {conf:>8.2f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
