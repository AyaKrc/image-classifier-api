from fastapi import APIRouter, UploadFile, File
from typing import List
import torch
from app.utils.preprocess import preprocess_image
from app.models.model_loader import load_model

router = APIRouter()

# load model once on startup
model, CLASS_NAMES, device = load_model()

@router.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    image = await file.read()
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return {"class": CLASS_NAMES[pred_class], "confidence": round(confidence, 4)}

@router.post("/predict/batch")
async def predict_batch(files: List[UploadFile]):
    results = []
    for file in files:
        image = await file.read()
        tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        results.append({
            "filename": file.filename,
            "class": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4)
        })
    return {"results": results}
