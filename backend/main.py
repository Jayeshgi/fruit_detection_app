"""
main.py
------------------------------------------------------------------------------
FastAPI Backend for the Fruit Detection Application.

Endpoints:
    POST /predict   - Upload a fruit image, get prediction + AI description
    GET  /health    - Health check endpoint

How to run:
    uvicorn main:app --reload --port 8000

Architecture:
    Image Upload -> PyTorch Model (fruit prediction) -> Gemini API (description) -> JSON Response
------------------------------------------------------------------------------
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from model_service import FruitDetector
from gemini_service import GeminiService

# ─── Global service instances (loaded once at startup) ────────────────────────
fruit_detector: FruitDetector = None
gemini_service: GeminiService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler — runs once when the server starts.
    Loads the PyTorch model and initializes the Gemini API.
    """
    global fruit_detector, gemini_service

    print("\n" + "=" * 60)
    print("  Fruit Detection API - Starting Up")
    print("=" * 60 + "\n")

    # Load PyTorch model
    try:
        fruit_detector = FruitDetector()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[ERROR] The server will start, but /predict will not work")
        print("[ERROR] until you train the model.\n")

    # Initialize Gemini
    try:
        gemini_service = GeminiService()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("[ERROR] The server will start, but AI descriptions will not work.\n")

    print("\n" + "=" * 60)
    print("  [OK] Server is ready!")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield  # Server is running

    # Cleanup on shutdown (if needed)
    print("\n[Server] Shutting down...")


# ─── Create FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Fruit Detection API",
    description="Upload a fruit image to get AI-powered prediction and description",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS (allow React frontend to call this API) ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Create React App
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check - verify the server and services are running."""
    return {
        "status": "healthy",
        "model_loaded": fruit_detector is not None,
        "gemini_ready": gemini_service is not None,
    }


@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    """
    Upload a fruit image and get:
    - The predicted fruit name and confidence score
    - An AI-generated description with nutrition info and health benefits

    Accepts: JPEG, PNG, WEBP images
    Returns: JSON with prediction and AI description
    """
    # ── Validate ──────────────────────────────────────────────────────────
    if fruit_detector is None:
        raise HTTPException(
            status_code=503,
            detail="PyTorch model is not loaded. Train the model first (python train.py)",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image (JPEG, PNG, WEBP).",
        )

    # ── Read image ────────────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # ── Step 1: PyTorch Prediction ────────────────────────────────────────
    try:
        # Multi-crop prediction handles centered/uncentered fruits
        predictions = fruit_detector.predict(image_bytes, top_k=5)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during model prediction: {str(e)}",
        )

    # ── Step 2: AI Refinement (The 'Master Judge') ────────────────────────
    # If Gemini is available, use Vision to double-check the image
    final_fruit_name = predictions[0]["name"]
    confidence = predictions[0]["confidence"]
    is_refined = False

    if gemini_service is not None:
        try:
            # Use Gemini Vision to pick the winner or fix the error
            final_fruit_name = await gemini_service.refine_prediction(
                image_bytes, predictions
            )
            is_refined = True
        except Exception as e:
            print(f"[Warning] Gemini refinement failed: {e}")

    # ── Step 3: Get Fruit Description ─────────────────────────────────────
    ai_description = None
    if gemini_service is not None:
        try:
            ai_description = await gemini_service.generate_description(
                final_fruit_name, use_retries=False
            )
        except Exception as e:
            print(f"[Warning] Gemini description failed: {e}")
            ai_description = gemini_service._get_fallback_data(final_fruit_name)

    # ── Build Response ────────────────────────────────────────────────────
    response = {
        "success": True,
        "prediction": {
            "fruit_name": final_fruit_name,
            "confidence": confidence if not is_refined else 100.0,
            "is_refined": is_refined,
            "pytorch_guesses": predictions,
        },
        "ai_description": ai_description,
    }

    return response


