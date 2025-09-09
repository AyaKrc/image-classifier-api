import time
import logging
import csv
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from app.routes import predict

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]  # logs show up in Hugging Face logs tab
)

app = FastAPI(title="AI-Generated Image Detector")

# include your routes
app.include_router(predict.router)

# --- Prepare CSV file for metrics ---
METRICS_FILE = "metrics.csv"
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "path", "latency_ms", "status_code"])


# --- Middleware to track every request ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logging.error(f"❌ Error while handling {request.url.path}: {e}")
        raise e
    finally:
        process_time = (time.time() - start_time) * 1000  # ms
        logging.info(f"📊 {request.method} {request.url.path} "
                     f"completed in {process_time:.2f}ms with {status_code}")

        # Save metrics to CSV
        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                request.url.path,
                f"{process_time:.2f}",
                status_code
            ])

    return response


# --- Homepage ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AI-Generated Image Detector</title>
        </head>
        <body style="font-family: Arial; margin: 2em;">
            <h1>🚀 Welcome to the AI-Generated Image Detector API</h1>
            <p>This API helps detect whether an image is AI-generated or real.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><b>Health Check:</b> <a href="/health">/health</a></li>
                <li><b>Prediction:</b> <code>/predict</code> (POST with an image file)</li>
            </ul>
            <p>Check logs and metrics in <code>metrics.csv</code> or the Hugging Face Logs tab.</p>
        </body>
    </html>
    """


@app.get("/health")
def health_check():
    return {"status": "ok"}
