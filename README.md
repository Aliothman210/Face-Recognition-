# 🎭 Face Recognition System

A comprehensive face recognition solution with two main functionalities: **face verification** (comparing two images) and **real-time face recognition** (recognizing faces from webcam against a database).

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Face Verification](#face-verification)
  - [Real-Time Face Recognition](#real-time-face-recognition)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [⚠️ Troubleshooting & Common Problems](#️-troubleshooting--common-problems)
- [License](#license)

## ✨ Features

### Face Verification (`Face_rec.py`)
- Compare two images to verify if they belong to the same person
- Uses VGG-Face deep learning model for accurate facial embeddings
- Calculates distance and similarity metrics
- Displays confidence percentage for matches
- Input validation and error handling for image files
- Side-by-side image visualization

### Real-Time Face Recognition (`main.py`)
- Live face recognition from webcam or IP camera
- Compare faces against a pre-built database
- Uses Facenet model for faster processing
- Cosine similarity matching algorithm
- Real-time bounding box and label display
- Adjustable similarity threshold
- Supports multiple images per person in database

## 📁 Project Structure

```
Face-Recognition-Project/
│
├── Face_rec.py              # Face verification script (1-to-1 comparison)
├── main.py                  # Real-time face recognition script
├── database/                # Database folder (required for main.py)
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── image3.jpg
│   └── person2/
│       ├── image1.jpg
│       └── image2.jpg
│
└── README.md               # This file
```

## 📦 Requirements

- Python 3.7 or higher
- OpenCV (cv2)
- DeepFace
- TensorFlow
- NumPy
- Matplotlib
- Webcam or IP camera (for real-time recognition)

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd Face-Recognition-Project
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Libraries
```bash
pip install opencv-python deepface tensorflow numpy matplotlib
```

Or install from `requirements.txt` if available:
```bash
pip install -r requirements.txt
```

### Step 4: Create Database Folder (for main.py)
```bash
mkdir database
# Then organize your images in subdirectories by person name
```

## 💻 Usage

### Face Verification

**Purpose**: Compare two images to check if they belong to the same person.

#### How to Run:
```bash
python Face_rec.py
```

#### Input:
The script will prompt you to enter paths to two images:
```
Face Recognition Verification System
========================================
Enter path to first image: path/to/image1.jpg
Enter path to second image: path/to/image2.jpg
```

#### Supported Image Formats:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

#### Output:
```
📊 Results:
Distance: 0.3456 (Threshold: 0.4500)
Are these the same person? Yes ✅
Confidence: 85.5%
```

#### Example Usage:
```bash
python Face_rec.py
Enter path to first image: photos/person_a_photo1.jpg
Enter path to second image: photos/person_a_photo2.jpg

# Shows both images side-by-side and displays verification result
```

---

### Real-Time Face Recognition

**Purpose**: Recognize faces in real-time from webcam or IP camera against a database.

#### Step 1: Prepare Your Database

Create a folder structure like this:
```
database/
├── Alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Bob/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Charlie/
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.jpg
```

**Tips for best results:**
- Use 2-5 clear photos per person
- Include photos from different angles
- Ensure face is clearly visible
- Use good lighting in photos
- All images should be in `.jpg`, `.jpeg`, or `.png` format

#### Step 2: Run the Script
```bash
python main.py
```

#### Step 3: Interact with the System
- The system will build embeddings from your database (this may take a few minutes)
- A window will open showing your webcam feed
- A green bounding box will appear in the center of the frame
- Recognized faces will be labeled with the person's name and confidence score
- Press **'q'** to quit

#### Output Example:
```
Building embeddings from database...
[OK] Processed photo1.jpg for Alice
[OK] Processed photo2.jpg for Alice
...
[INFO] Built embeddings for 3 people
Starting face recognition... Press 'q' to quit
```

#### Camera Configuration:

**Use Default Webcam:**
```python
capture = cv2.VideoCapture(0)  # Already set in code
```

**Use IP Webcam:**
If you want to use an IP camera instead, uncomment and modify:
```python
capture = cv2.VideoCapture("http://192.168.1.22:8080/video")  # Change IP to yours
```

---

## 🗄️ Database Setup

### Creating an Effective Database:

1. **Create the database folder** in your project root:
   ```bash
   mkdir database
   ```

2. **Add subfolders for each person**:
   ```bash
   mkdir database/person_name
   ```

3. **Add photos to each person's folder**:
   - Minimum 2-3 photos per person recommended
   - Maximum files: unlimited (but more = slower processing)
   - Formats: `.jpg`, `.jpeg`, `.png`

4. **Example structure**:
   ```
   database/
   ├── John_Doe/
   │   ├── 1.jpg
   │   ├── 2.jpg
   │   └── 3.jpg
   └── Jane_Smith/
       ├── 1.jpg
       └── 2.jpg
   ```

### Best Practices:
- Use clear, frontal face photos
- Include different lighting conditions
- Include different angles if possible
- Avoid heavy makeup or accessories changes
- Use high-resolution images (at least 200x200 pixels)
- Keep file names simple (numbers or simple descriptors)

---

## ⚙️ Configuration

### Adjusting Similarity Threshold (main.py)

In `main.py`, find this line:
```python
if max_similarity > 0.4:  # Adjust this value based on testing
```

- **Lower threshold (e.g., 0.3)**: More lenient, may have false positives
- **Higher threshold (e.g., 0.6)**: More strict, may miss some recognitions
- **Default (0.4)**: Balanced setting

### Models Used

**Face Verification (Face_rec.py):**
- Model: **VGG-Face**
- Best for: High accuracy in face verification tasks

**Real-Time Recognition (main.py):**
- Model: **Facenet**
- Best for: Faster processing in real-time applications

---

## ⚠️ Troubleshooting & Common Problems

### Problem 1: DeepFace Model Download Failures

**Error Messages:**
- `ChunkedEncodingError`
- `Model could not be found`
- `Connection timeout`
- `URLError: urlopen error`

**Solution:**

DeepFace models might fail to download automatically due to network issues. You can manually download the required models:

#### For Face Verification (VGG-Face Model):

1. Download from:
   ```
   https://github.com/serengil/deepface_models/releases/download/v1.0/vggface_weights.h5
   ```

2. Create the weights directory:
   ```
   Windows: C:\Users\<YourUsername>\.deepface\weights\
   macOS/Linux: ~/.deepface/weights/
   ```

3. Save the file as `vggface_weights.h5` in the weights folder

#### For Real-Time Recognition (Facenet Model):

1. Download from:
   ```
   https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
   ```

2. Save it to the same weights directory (created above)

3. Save the file as `facenet_weights.h5`

#### After placing files:
After placing the model files in the weights folder, DeepFace will load them directly without attempting to re-download.

**Alternative: Clear Cache and Re-download**
```python
# In Python, before importing DeepFace:
import os
import shutil

# Remove cached models
cache_dir = os.path.expanduser("~/.deepface")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Then run your script again
```

---

### Problem 2: Camera/Webcam Not Opening

**Error Message:** `Error: Could not open camera stream.`

**Solutions:**
- Make sure your webcam is connected and not in use by another application
- Try using a different camera index (change `0` to `1` or `2` in `cv2.VideoCapture()`)
- For IP webcam, verify the IP address is correct
- Check camera permissions on your system (especially on macOS/Linux)
- Restart your system if camera is still not recognized

---

### Problem 3: Poor Face Recognition Accuracy

**Symptoms:**
- Faces not being recognized
- Wrong person identified
- Constant "Unknown" output

**Solutions:**
- **Improve database quality**: Add more diverse photos (different angles, lighting)
- **Adjust threshold**: Lower the similarity threshold in `main.py` (change `0.4` to `0.3`)
- **Better photos**: Ensure faces are clear and well-lit in database images
- **Face size**: Make sure faces occupy at least 50% of the image frame
- **Distance**: Stand at a reasonable distance from the camera (1-2 meters)
- **Lighting**: Ensure good lighting conditions when using the system

---

### Problem 4: "No embeddings found" Error

**Error Message:** `[ERROR] No embeddings found! Make sure your database folder exists and contains images.`

**Solutions:**
- Verify the `database` folder exists in your project root
- Check that the database folder contains subfolders (one per person)
- Ensure subfolders contain image files (`.jpg`, `.jpeg`, `.png`)
- Verify image files are not corrupted
- Check file permissions allow reading the images

**Example valid structure:**
```
your_project/
├── main.py
├── Face_rec.py
└── database/
    ├── Alice/
    │   └── photo1.jpg
    └── Bob/
        └── photo1.jpg
```

---

### Problem 5: Script Runs Slowly / High CPU Usage

**Causes:**
- Too many images in database
- Image resolution too high
- Frame size not optimized

**Solutions:**
- Reduce images per person (keep 2-4 per person)
- Use lower resolution images (800x600 or smaller)
- Reduce frame resolution in real-time recognition
- Close other applications consuming CPU
- Use GPU if available (install `tensorflow-gpu`)

---

### Problem 6: TensorFlow/CUDA Warnings

**Warnings about GPU, CUDA, cuDNN**

**Note:** These are warnings, not errors. The system will still work, but here are ways to suppress them:

```python
# Add at the very beginning of your script:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore")
```

To use GPU (if available):
```bash
pip install tensorflow-gpu  # For GPU support
pip install tensorflow-macos  # For macOS
```

---

### Problem 7: OpenCV Window Not Displaying (Linux/Mac)

**Solution:**
```python
# Add this before cv2.imshow():
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 640, 480)
```
(This is already in the code)

---

### Problem 8: "enforce_detection=False" Warning

**What it means:** The system won't fail if no face is detected in an image

**Why it's there:** Some images might have subtle faces that need lenient detection

**If you want strict detection**, change in `main.py`:
```python
enforce_detection=True  # Will skip frames with no detected faces
```


---

## 🤝 Contributing

Contributions are welcome! Feel free to fork, improve, and submit pull requests.

---

## 📧 Support

If you encounter any issues not listed in the troubleshooting section, please check:
1. DeepFace documentation: https://github.com/serengil/deepface
2. OpenCV documentation: https://docs.opencv.org/
3. TensorFlow documentation: https://www.tensorflow.org/

---

## 🎯 Future Improvements

Potential enhancements for this project:
- Add face detection and tracking visualization
- Implement multiple face recognition in single frame
- Add face encoding storage for faster comparison
- Create a GUI interface
- Add facial expression detection
- Implement face verification API endpoint
- Add logging and statistics

---

**Last Updated:** 2025
**Version:** 1.0
