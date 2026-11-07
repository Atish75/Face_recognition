#!/usr/bin/env python3
"""
Face Recognition Web Application (FastAPI + HTML5/JavaScript)
--------------------------------------------------------------
Features:
- Browser-based camera access via HTML5 getUserMedia API (PC, Mobile, Moto Phone).
- Base64 image decoding and 128-d face encoding extraction.
- RESTful endpoints:
    GET  /           -> Serves interactive HTML5 camera UI
    POST /recognize  -> Accepts frame, returns detected faces, boxes & confidence
    POST /register   -> Enrolls a new person's face into 'known_faces/'
    GET  /faces      -> Lists all currently registered faces
    DELETE /faces/{name} -> Removes a registered person
- Cloud-ready for Render, Railway, Docker, or local execution.
"""

import os
import io
import re
import base64
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_rec_app")

try:
    import face_recognition
except ImportError:
    face_recognition = None
    logger.warning("face_recognition library not installed. Install via pip install face_recognition.")

app = FastAPI(title="Face Recognition Web API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# In-memory storage for known face encodings and names
known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []
TOLERANCE = 0.55  # Distance threshold: lower = stricter matching


def load_known_faces() -> int:
    """
    Scans KNOWN_FACES_DIR and loads 128-d encodings for all known pictures.
    """
    global known_face_encodings, known_face_names
    if face_recognition is None:
        return 0

    encodings = []
    names = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    logger.info(f"Scanning '{KNOWN_FACES_DIR}' for registered faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_exts:
            continue

        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        person_name = os.path.splitext(filename)[0].replace("_", " ").title()

        try:
            image = face_recognition.load_image_file(filepath)
            img_encs = face_recognition.face_encodings(image)
            if len(img_encs) > 0:
                encodings.append(img_encs[0])
                names.append(person_name)
                logger.info(f"Loaded face: {person_name} ({filename})")
            else:
                logger.warning(f"No face detected in file: {filename}")
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    known_face_encodings = encodings
    known_face_names = names
    logger.info(f"Total {len(known_face_names)} faces loaded into memory.")
    return len(known_face_names)


def decode_base64_image(image_data: str) -> np.ndarray:
    """
    Decodes data:image/jpeg;base64,... string into an OpenCV RGB image array.
    """
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        binary_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(binary_data, np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("cv2.imdecode returned None")
        # Convert BGR to RGB for face_recognition library
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.on_event("startup")
def startup_event():
    load_known_faces()


# -----------------------------
# Pydantic Request Models
# -----------------------------
class RecognizeRequest(BaseModel):
    image: str  # Data URL (data:image/jpeg;base64,...)
    tolerance: Optional[float] = 0.55


class RegisterRequest(BaseModel):
    name: str
    image: str  # Data URL


# -----------------------------
# API Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """
    Renders the browser-based camera HTML5 application.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "registered_count": len(known_face_names)},
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "face_recognition (dlib 128-d)",
        "known_faces_count": len(known_face_names),
    }


@app.get("/faces")
async def list_faces():
    """
    Returns list of all registered names and their image files.
    """
    faces = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for filename in os.listdir(KNOWN_FACES_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            faces.append({"name": name, "filename": filename})
    return {"success": True, "faces": faces, "total": len(faces)}


@app.delete("/faces/{name}")
async def delete_face(name: str):
    """
    Deletes a registered face photo and refreshes memory.
    """
    clean_target = name.strip().replace(" ", "_").lower()
    deleted = False

    for filename in os.listdir(KNOWN_FACES_DIR):
        file_base = os.path.splitext(filename)[0].lower()
        if file_base == clean_target:
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                os.remove(filepath)
                deleted = True
                break
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found in database.")

    load_known_faces()
    return {"success": True, "message": f"Successfully deleted {name}."}


@app.post("/register")
async def register_face(payload: RegisterRequest):
    """
    Registers a new person by saving their image and extracting encodings.
    """
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="face_recognition library not available.")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    rgb_image = decode_base64_image(payload.image)

    # Detect faces to ensure good sample
    locations = face_recognition.face_locations(rgb_image)
    if len(locations) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Photo me koi chehra detect nahi hua. Roshni me dubara photo lein."},
        )
    elif len(locations) > 1:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Ek se zyada chehre dikh rahe hain. Sirf ek vyakti hona chahiye."},
        )

    # Save image as clean file
    clean_filename = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())}.jpg"
    target_path = os.path.join(KNOWN_FACES_DIR, clean_filename)

    # Convert back to BGR for cv2 saving
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(target_path, bgr_image)

    # Reload encodings
    load_known_faces()

    return {
        "success": True,
        "message": f"'{name}' successfully register ho gaye!",
        "filename": clean_filename,
        "total_known": len(known_face_names),
    }


@app.post("/recognize")
async def recognize_faces(payload: RecognizeRequest):
    """
    Receives video frame from browser, detects all faces, and matches against known database.
    """
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="face_recognition library not installed.")

    rgb_image = decode_base64_image(payload.image)
    h, w = rgb_image.shape[:2]

    # Downscale frame for fast detection (0.5x scale)
    scale = 0.5
    small_rgb = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)

    face_locations = face_recognition.face_locations(small_rgb)
    if len(face_locations) == 0:
        return {"success": True, "faces": [], "image_size": {"width": w, "height": h}}

    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

    tol = payload.tolerance or TOLERANCE
    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back to original frame size
        inv = 1.0 / scale
        orig_box = {
            "top": int(top * inv),
            "right": int(right * inv),
            "bottom": int(bottom * inv),
            "left": int(left * inv),
            "width": int((right - left) * inv),
            "height": int((bottom - top) * inv),
        }

        name = "Unknown"
        confidence = 0.0

        if len(known_face_encodings) > 0:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist <= tol:
                name = known_face_names[best_idx]
                confidence = round(max(0.0, min(100.0, (1.0 - (best_dist / tol)) * 100)), 1)

        results.append({
            "name": name,
            "confidence": confidence,
            "box": orig_box,
        })

    return {
        "success": True,
        "faces": results,
        "total_detected": len(results),
        "image_size": {"width": w, "height": h},
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)