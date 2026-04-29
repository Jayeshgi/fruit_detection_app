# FruitLens - Fruit Detection App

A full-stack fruit detection application that identifies fruits from images and provides nutritional information.

## Project Structure

```
fruit_detection_app/
├── model/                    # PyTorch ML Model
│   ├── config.py             # Hyperparameters & paths
│   ├── dataset.py            # Data loading & augmentation
│   ├── download_dataset.py   # Kaggle dataset downloader
│   ├── fruit_model.py        # MobileNetV3 architecture
│   ├── train.py              # Training pipeline
│   ├── predict.py            # CLI prediction script
│   ├── requirements.txt      # Python dependencies
│   ├── saved_models/         # Trained model weights (best_fruit_model.pth)
│   ├── data/                 # Fruits-360 dataset
│   └── venv/                 # Python virtual environment
│
├── backend/                  # FastAPI Backend
│   ├── main.py               # API endpoints (/predict, /health)
│   ├── model_service.py      # PyTorch model loader & inference
│   ├── gemini_service.py     # Gemini AI descriptions + fallback data
│   ├── .env                  # Gemini API key (DO NOT share)
│   ├── requirements.txt      # Python dependencies
│   └── test_api.py           # Quick API test script
│
├── frontend/                 # React Frontend (Vite)
│   ├── src/
│   │   ├── App.jsx           # Main app component
│   │   ├── App.css           # Component styles
│   │   ├── index.css         # Global styles & design system
│   │   └── main.jsx          # Entry point
│   ├── index.html            # HTML template
│   └── package.json          # Node dependencies
│
└── README.md                 # This file
```

## How to Run

### 1. Start the Backend (Terminal 1)

```bash
cd c:\Users\erjay\OneDrive\Desktop\fruit_detection_app\backend
c:\Users\erjay\OneDrive\Desktop\fruit_detection_app\model\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

API will be at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Start the Frontend (Terminal 2)

```bash
cd c:\Users\erjay\OneDrive\Desktop\fruit_detection_app\frontend
npm run dev
```

App will be at: http://localhost:5173

## Current Status

- [x] PyTorch model trained (90.6% accuracy, 257 fruit classes)
- [x] FastAPI backend with /predict endpoint
- [x] Gemini AI integration with retry logic + fallback nutritional data
- [x] React frontend with drag-and-drop upload

## TODO

- [ ] Improve model accuracy (more epochs, unfreeze backbone)
- [ ] Speed up Gemini responses (add caching)
- [ ] Polish UI (remove AI branding, make it more personal)
- [ ] Deploy (optional)

## Tech Stack

- **ML Model**: PyTorch + MobileNetV3 (transfer learning)
- **Dataset**: Fruits-360 (Kaggle) — 257 classes, 180k images
- **Backend**: FastAPI + Uvicorn
- **AI Descriptions**: Google Gemini 2.0 Flash
- **Frontend**: React + Vite
