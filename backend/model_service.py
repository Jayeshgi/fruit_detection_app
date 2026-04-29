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
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms

# Add model directory to path so we can import fruit_model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from PIL import Image
from io import BytesIO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.num_classes = 0

        # BACK TO BASICS: Direct resize to 224x224. 
        # For Fruits-360, this is often better as it preserves the whole fruit.
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the trained model checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        print(f"[ModelService] Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]

        # Recreate exact architecture using our fruit_model factory
        from fruit_model import create_model
        self.model = create_model(num_classes=self.num_classes, freeze_backbone=False)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ModelService] Loaded {self.num_classes} classes on {self.device}")

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict]:
        """
        Multi-Crop Inference: Looks at the image in 3 different ways 
        and picks the one the model is most confident about.
        """
        try:
            raw_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Create 3 different ways to look at the image
            crops = [
                # 1. Full Image (squished)
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(raw_image),
                # 2. Center Crop (zoomed in)
                transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)])(raw_image),
                # 3. Tight Center Crop (very zoomed)
                transforms.Compose([transforms.Resize(300), transforms.CenterCrop(IMAGE_SIZE)])(raw_image)
            ]

            best_predictions = None
            max_confidence = -1.0

            for crop in crops:
                tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])(crop).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    
                    # If this "view" of the fruit is more confident, keep it
                    if conf.item() > max_confidence:
                        max_confidence = conf.item()
                        # Get Top K for this best view
                        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
                        
                        best_predictions = []
                        for i in range(top_probs.size(1)):
                            p = top_probs[0][i].item()
                            name = self.class_names[top_indices[0][i].item()]
                            import re
                            clean = re.sub(r"\s*\d+\s*$", "", name).strip()
                            best_predictions.append({"name": clean, "confidence": round(p * 100, 2)})

            # Logging
            if best_predictions:
                print(f"[Multi-Crop] Best: {best_predictions[0]['name']} ({best_predictions[0]['confidence']}%)")

            return best_predictions

        except Exception as e:
            print(f"[Predict Error] {e}")
            return [{"name": "Error", "confidence": 0.0}]



