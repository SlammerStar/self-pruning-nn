"""
api.py
======
FastAPI inference service for the trained Self-Pruning Neural Network.

Endpoints
---------
  GET  /health          — liveness probe
  GET  /model/info      — architecture + gate statistics for a given λ
  POST /predict         — classify a CIFAR-10 image, return class + sparsity %

Start the server
----------------
    pip install fastapi uvicorn python-multipart pillow
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Example request
---------------
    curl -X POST "http://localhost:8000/predict" \\
         -F "file=@cat.png" \\
         -F "lambda_val=0.01"
"""

import io
import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from models.network import SelfPruningNetwork

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)),
])

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("./outputs/checkpoints")

app = FastAPI(
    title="Self-Pruning Neural Network API",
    description=(
        "Classifies 32×32 CIFAR-10 images using a self-pruning MLP. "
        "Accepts any image size — resized internally to 32×32."
    ),
    version="1.0.0",
)

_cache: dict[float, SelfPruningNetwork] = {}


def _get_model(lam: float) -> SelfPruningNetwork:
    if lam not in _cache:
        ckpt = CHECKPOINT_DIR / f"best_lambda_{lam}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}. "
                "Run  python main.py  first to train and save checkpoints."
            )
        model = SelfPruningNetwork(input_dim=3072, hidden_dims=[1024, 512, 256], num_classes=10)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()
        _cache[lam] = model
        log.info(f"Loaded checkpoint  λ={lam}")
    return _cache[lam]


# ── Response schemas ──────────────────────────────────────────────────────────

class HealthResp(BaseModel):
    status: str
    device: str

class PredictResp(BaseModel):
    predicted_class:  str
    class_index:      int
    confidence:       float
    sparsity_percent: float
    lambda_val:       float
    param_counts:     dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResp, tags=["System"])
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/model/info", tags=["Model"])
async def model_info(lambda_val: float = 0.01):
    """Return architecture details, gate statistics, and sparsity for a λ checkpoint."""
    try:
        model = _get_model(lambda_val)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    gates = model.all_gates().numpy()
    return {
        "lambda":          lambda_val,
        "architecture":    {"input_dim": model.input_dim,
                            "hidden_dims": model.hidden_dims,
                            "num_classes": model.num_classes},
        "param_counts":    model.param_counts(),
        "sparsity_percent": round(model.sparsity() * 100, 2),
        "gate_mean":        round(float(gates.mean()), 4),
        "gate_min":         round(float(gates.min()),  4),
        "gate_max":         round(float(gates.max()),  4),
    }


@app.post("/predict", response_model=PredictResp, tags=["Inference"])
async def predict(
    file:       UploadFile = File(...,  description="PNG or JPEG image"),
    lambda_val: float      = Form(0.01, description="λ of the checkpoint to use"),
):
    """
    Classify an image.

    - Upload any PNG/JPEG — it will be resized to 32×32 internally.
    - `lambda_val` selects which trained checkpoint to use (must exist in outputs/checkpoints/).
    - Returns the predicted CIFAR-10 class, confidence, and the model's current sparsity.
    """
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Invalid image: {e}")

    try:
        model = _get_model(lambda_val)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs     = torch.softmax(model(tensor), dim=1)[0]
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

    return PredictResp(
        predicted_class=CLASSES[class_idx],
        class_index=class_idx,
        confidence=round(confidence, 4),
        sparsity_percent=round(model.sparsity() * 100, 2),
        lambda_val=lambda_val,
        param_counts=model.param_counts(),
    )
