# Face Recognition AI - Web Platform

Full-stack real-time Face Recognition web application powered by **FastAPI**, **HTML5/JavaScript (WebRTC)**, and **128-dimensional Deep Face Encodings**.

## Features
- **Browser Camera Access**: Runs on any device (PC, Laptop, Android phone, iPhone) via standard browser camera permissions.
- **Real-Time Recognition**: Detects faces in video frames, draws bounding boxes, tags names, and calculates match confidence.
- **In-Browser Face Enrollment**: Type a name, take a snapshot, and register faces instantly into the AI database.
- **Cloud-Ready**: Prepared with `Dockerfile` and `requirements.txt` for 1-click deployment on **Render**, **Railway**, or Docker hosts.

---

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

## 2. GitHub & Render Deployment Guide

### Step 1: Push to your GitHub Private Repo
```bash
git init
git add .
git commit -m "Initial commit: Face Recognition Web App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step 2: Deploy on Render (Free Tier)
1. Login to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Settings:
   - **Name**: `face-recognition-app` (or any name you like)
   - **Language / Runtime**: Choose **Docker** *(Recommended: Docker automatically handles `dlib` & `cmake` compilation cleanly)*.
   - **Instance Type**: Free
5. Click **Create Web Service**.

Once built, Render will provide a live HTTPS link (e.g. `https://face-recognition-app.onrender.com`).  
Open that link on your phone or laptop browser, allow camera permissions, and start recognizing faces in real-time!
