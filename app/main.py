from fastapi import FastAPI
from app.routes import predict

app = FastAPI(title="AI-Generated Image Detector")

app.include_router(predict.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
