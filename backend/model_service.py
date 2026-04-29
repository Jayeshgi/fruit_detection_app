"""
model_service.py
──────────────────────────────────────────────────────────────────────────────
Handles loading the trained PyTorch model and running inference.

This is a standalone module that does NOT import from the model/ directory.
Instead, it re-implements the minimal code needed to load and run the model,
so the backend is self-contained.
──────────────────────────────────────────────────────────────────────────────
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO

# ─── Constants (must match what was used during training) ─────────────────────
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Path to the trained model (relative to this file's location)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "model", "saved_models", "best_fruit_model.pth",
)


class FruitDetector:
    """
    Loads the trained PyTorch model once and provides a predict() method.
    Designed to be initialized once when the FastAPI server starts.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.num_classes = 0

        # Image preprocessing — MUST match test transforms used during training
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the trained model checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please run 'python train.py' in the model/ directory first."
            )

        print(f"[ModelService] Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]

        # Recreate the model architecture
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, self.num_classes)

        # Load trained weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[ModelService] Model loaded successfully!")
        print(f"[ModelService]   Classes: {self.num_classes}")
        print(f"[ModelService]   Device : {self.device}")
        print(f"[ModelService]   Accuracy: {checkpoint.get('test_acc', 'N/A')}%")

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict]:
        """
        Predict the fruit from raw image bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            top_k:       Number of top predictions to return.

        Returns:
            List of dicts: [{"name": "Apple", "confidence": 97.32}, ...]
        """
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(
                probabilities, k=min(top_k, self.num_classes)
            )

        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                "name": self.class_names[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            })

        return predictions
