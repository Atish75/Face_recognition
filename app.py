#!/usr/bin/env python3
"""
Face Recognition Web Application (FastAPI + HTML5/JavaScript)
--------------------------------------------------------------
Clean, lightweight server that automatically serves index.html
from root or templates/ folder, processes frames, and matches faces.
"""

import os
import re
import base64
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_rec_app")

try:
    import face_recognition
except ImportError:
    face_recognition = None
    logger.warning("face_recognition library not installed.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []
TOLERANCE = 0.55


def load_known_faces() -> int:
    global known_face_encodings, known_face_names
    if face_recognition is None:
        return 0

    encodings = []
    names = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

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
                logger.info(f"Loaded face: {person_name}")
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    known_face_encodings = encodings
    known_face_names = names
    return len(known_face_names)


def decode_base64_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        binary_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(binary_data, np.uint8)
        bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("Failed to decode image")
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_known_faces()
    yield


app = FastAPI(title="Face Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecognizeRequest(BaseModel):
    image: str
    tolerance: Optional[float] = 0.55


class RegisterRequest(BaseModel):
    name: str
    image: str


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """
    Checks for index.html in root directory or templates/ folder.
    """
    possible_paths = [
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(BASE_DIR, "templates", "index.html"),
        "index.html",
        "templates/index.html",
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception as e:
                logger.error(f"Error reading {p}: {e}")

    return HTMLResponse(
        "<h2>index.html nahi mila! Kripya GitHub me index.html upload karein.</h2>",
        status_code=404,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "known_faces_count": len(known_face_names),
    }


@app.get("/faces")
async def list_faces():
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
    clean_target = name.strip().replace(" ", "_").lower()
    deleted = False

    for filename in os.listdir(KNOWN_FACES_DIR):
        if os.path.splitext(filename)[0].lower() == clean_target:
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                os.remove(filepath)
                deleted = True
                break
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found")

    load_known_faces()
    return {"success": True, "message": f"Deleted {name}"}


@app.post("/register")
async def register_face(payload: RegisterRequest):
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="face_recognition library missing")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name required")

    rgb_image = decode_base64_image(payload.image)
    locations = face_recognition.face_locations(rgb_image)

    if len(locations) == 0:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Koi chehra detect nahi hua."},
        )
    elif len(locations) > 1:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Ek se zyada chehre dikh rahe hain."},
        )

    clean_filename = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())}.jpg"
    target_path = os.path.join(KNOWN_FACES_DIR, clean_filename)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(target_path, bgr_image)
    load_known_faces()

    return {
        "success": True,
        "message": f"'{name}' register ho gaye!",
        "filename": clean_filename,
        "total_known": len(known_face_names),
    }


@app.post("/recognize")
async def recognize_faces(payload: RecognizeRequest):
    if face_recognition is None:
        raise HTTPException(status_code=500, detail="face_recognition library missing")

    rgb_image = decode_base64_image(payload.image)
    h, w = rgb_image.shape[:2]

    scale = 0.5
    small_rgb = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)

    face_locations = face_recognition.face_locations(small_rgb)
    if len(face_locations) == 0:
        return {"success": True, "faces": [], "image_size": {"width": w, "height": h}}

    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)
    tol = payload.tolerance or TOLERANCE
    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
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
    uvicorn.run(app, host="0.0.0.0", port=port)
