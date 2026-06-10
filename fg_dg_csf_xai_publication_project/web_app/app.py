import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web_app.inference import WebDetector, PROJECT_ROOT


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"
TEMPLATE_DIR = APP_DIR / "templates"
ARTIFACT_DIR = APP_DIR / "artifacts"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FG-DG-CSF-XAI Deepfake Image Detector",
    description="Live real/fake image prediction with temperature calibration, Grad-CAM and FFT explanations.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

_detector = None


def get_detector():
    global _detector

    if _detector is None:
        checkpoint_path = os.environ.get("MODEL_CKPT")
        temperature_path = os.environ.get("TEMPERATURE_JSON", str(ARTIFACT_DIR / "temperature.json"))
        calibration_path = os.environ.get("CALIBRATION_JSON", str(ARTIFACT_DIR / "calibration_metrics.json"))

        _detector = WebDetector(
            checkpoint_path=checkpoint_path,
            temperature_path=temperature_path,
            calibration_metrics_path=calibration_path,
            static_output_dir=STATIC_DIR / "results",
        )

    return _detector


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    detector_status = {}
    try:
        detector = get_detector()
        detector_status = {
            "loaded": True,
            "checkpoint": str(detector.checkpoint_path),
            "temperature": detector.temperature,
            "model_ece": detector.calibration_metrics.get("ece", detector.calibration_metrics.get("ece_after_temperature")),
            "model_brier": detector.calibration_metrics.get("brier", detector.calibration_metrics.get("brier_after_temperature")),
        }
    except Exception as e:
        detector_status = {
            "loaded": False,
            "error": str(e),
        }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "detector": detector_status,
        },
    )


@app.get("/health")
async def health():
    try:
        detector = get_detector()
        return {
            "status": "ok",
            "checkpoint": str(detector.checkpoint_path),
            "temperature": detector.temperature,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e),
            },
        )


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    known_label: str = Form("unknown"),
):
    if image.content_type is not None and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(image.filename or "upload.png").suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
        suffix = ".png"

    upload_path = UPLOAD_DIR / f"upload_{os.urandom(6).hex()}{suffix}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        detector = get_detector()
        detector.reload_temperature()
        result = detector.predict(
            upload_path,
            known_label=known_label,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model():
    global _detector
    _detector = None
    detector = get_detector()
    return {
        "status": "reloaded",
        "checkpoint": str(detector.checkpoint_path),
        "temperature": detector.temperature,
    }
