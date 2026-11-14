"""
Moon Vision - FastAPI backend.

Run:   uvicorn app:app --reload --port 8000      (from the backend/ folder)
Open:  http://localhost:8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "moon_cv" imports work when run as backend.app (Vercel)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from moon_cv.analyze import analyze, get_cnn

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Moon Vision API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    cnn = get_cnn()
    return {"ok": True, "cnn_loaded": cnn.available, "cnn": cnn.info if cnn.available else None}


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    hemisphere: str = Form("north"),          # north | south
    # Exposure settings read from the photo's EXIF in the browser. Sent separately because
    # the browser may shrink the photo before upload (Vercel's 4.5 MB limit), dropping EXIF.
    exposure_time: float | None = Form(None),
    f_number: float | None = Form(None),
    iso: float | None = Form(None),
):
    if hemisphere not in ("north", "south"):
        raise HTTPException(400, "hemisphere must be north or south")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 25 MB")
    try:
        client_exif = {"exposure_time": exposure_time, "f_number": f_number, "iso": iso}
        return analyze(data, hemisphere=hemisphere, client_exif=client_exif)
    except Exception as e:  # unreadable image etc.
        raise HTTPException(422, f"could not analyse image: {e}")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
