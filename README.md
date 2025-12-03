# 🎭 Face Recognition System

A practical face recognition project featuring **1-to-1 face verification** and **real-time recognition** using **DeepFace (VGG-Face / Facenet)** and **OpenCV**.  
The focus is on solving real issues encountered during development (model downloads, TensorFlow errors, DeepFace failures).

---



## ✨ Features

- Face verification using **VGG-Face**
- Real-time recognition using **Facenet**
- Cosine similarity matching
- Multi-image database per person
- Adjustable threshold
- Webcam or IP camera support
- Clean folder structure

---

## 📁 Project Structure

```
Face-Recognition/
├── Face_rec.py
├── main.py
├── requirements.txt
├── database/
│ ├── person1/
│ └── person2/
└── README.md
```

---

## 🔧 Installation

Install all required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Face Verification (image vs image)

```bash
python Face_rec.py
```

Enter two image paths → the system compares them and shows confidence.

### 🔹 Real-Time Face Recognition

```bash
python main.py
```

Works with:
- Laptop webcam
- IP camera: `http://YOUR_IP:8080/video`

---

## 🧠 How It Works

1. Detect the face
2. Generate embeddings (VGG-Face or Facenet)
3. Compare with cosine similarity
4. Match if similarity > threshold
5. Draw bounding box + label

---

## 🔥 Problems I Actually Faced (and how I solved them)

### ❗ 1) DeepFace wouldn't download models

**Issues:**
- Timeout
- "ChunkedEncodingError"
- Corrupted downloads
- GitHub hosting errors

**Fix:**
- Downloaded the models manually
- Placed them in the DeepFace weights directory:

```
Windows: C:/Users/<USER>/.deepface/weights/
Linux/Mac: ~/.deepface/weights/
```

- Matching filenames → DeepFace loads them instantly.

### ❗ 2) Facenet model fails to load / crashes TensorFlow

**Fix:**
- Manual download
- Deleted corrupted cache:

```bash
~/.deepface
```

- Relaunched → worked normally.

### ❗ 3) First-run extremely slow

**Reason:** DeepFace builds embeddings for all database images.

**Fix:**
- Reduced images per person to 2–4
- Downscaled database images before embedding
- Speed improved drastically

### ❗ 4) Webcam freezing / lag

**Fix:** Set lower resolution in main.py:

```python
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### ❗ 5) Recognition accuracy drops with poor lighting

**Fix:**
- Added better database images
- Adjusted threshold
- Ensured frontal faces in database

---

## 🔤 Accuracy Notes

- VGG-Face → best for verification
- Facenet → best for real-time
- Ideal threshold: 0.35–0.45
- Accuracy heavily depends on database image quality

---

## ⚠️ Common Issues (short & useful)

### "No embeddings found"
Database folder empty or wrong path.

### "Unknown" constantly
Bad lighting or faces too small.

### Camera not working
Another program is using the webcam.

### TensorFlow GPU warnings
Ignore — CPU works fine.

---

## 🚀 Future Improvements

- Multi-face tracking
- Faster caching
- API endpoint
- Expression recognition
- Logging + stats reporting

---

## ✔️ Status

Clean • Fast • Practical • Internship-ready
