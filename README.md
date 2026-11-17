# Face Recognition AI - Web Platform

Full-stack real-time Face Recognition web application powered by **FastAPI**, **HTML5/JavaScript (WebRTC)**, and **128-dimensional Deep Face Encodings**.

## Features
- **Browser Camera Access**: Runs on any device (PC, Laptop, Android phone, iPhone) via standard browser camera permissions.
- **Real-Time Recognition**: Detects faces in video frames, draws bounding boxes, tags names, and calculates match confidence.
- **In-Browser Face Enrollment**: Type a name, take a snapshot, and register faces instantly into the AI database.
- **Cloud-Ready**: Prepared with `Dockerfile` and `requirements.txt` for 1-click deployment on **Render**, or Docker hosts.

---
'''
## 🔗 Live Link :- https://face-recogitiion.onrender.com
It is deployed and works on any System (Pc, Laptop, Mobile)
'''
## Project Structure
```text
├── app.py                  # FastAPI Backend API & Recognition Engine
├── templates/
│   └── index.html          # HTML5 + TailwindCSS + JavaScript WebRTC Frontend
├── known_faces/            # Folder storing registered face images (.jpg/.png)
├── requirements.txt        # Python library dependencies
├── Dockerfile              # Docker container setup (for Render / Cloud)
├── .gitignore
└── README.md
```

---

## 1. Local Testing
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run FastAPI local server
python app.py
```
Open your browser at `http://localhost:8000`.

---

