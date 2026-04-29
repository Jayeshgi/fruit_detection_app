"""
Configuration for the Fruit Detection Model.
All hyperparameters and paths are defined here so they are easy to tweak.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "fruits-360")  # will be created by download script
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")
CLASS_NAMES_FILE = os.path.join(MODEL_SAVE_PATH, "class_names.txt")

# ─── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224          # MobileNetV3 expects 224x224
BATCH_SIZE = 128          # Lowered for stability on Laptop GPU
NUM_WORKERS = 6           # Optimized for Windows

# ─── Training Hyperparameters ─────────────────────────────────────────────────
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001      # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4         # L2 regularization

# ─── ImageNet Normalization (used by all torchvision pretrained models) ───────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "resnet18"   # Options: mobilenet_v3_small, mobilenet_v3_large, resnet18
FREEZE_BACKBONE = False             # Fine-tune ALL layers for higher accuracy
