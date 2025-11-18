# main.py
"""Main script to run face recognition using webcam and a pre-built database of faces.
 It captures video from the webcam (or IP Webcam), processes each frame to recognize faces,
 and displays the results in real-time.
"""


import cv2
import os
import numpy as np
from deepface import DeepFace
import warnings
warnings.filterwarnings("ignore")

# Global variable to store embeddings
embeddings = {}

# ----------------- Prepare Embeddings -----------------
def build_embeddings(database_path="database"):
    global embeddings
    embeddings = {}
    
    if not os.path.exists(database_path):
        print(f"[ERROR] Database path '{database_path}' does not exist!")
        return embeddings
    
    for person_name in os.listdir(database_path):
        person_folder = os.path.join(database_path, person_name)
        if not os.path.isdir(person_folder):
            continue  # Skip non-folder files
            
        person_embeddings = []
        for img_file in os.listdir(person_folder):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files
                
            img_path = os.path.join(person_folder, img_file)
            try:
                result = DeepFace.represent(
                    img_path, 
                    model_name="Facenet",
                    enforce_detection=False  # Allow processing even if face detection fails
                )
                if result:
                    embedding = result[0]["embedding"]
                    person_embeddings.append(embedding)
                    print(f"[OK] Processed {img_file} for {person_name}")
            except Exception as e:
                print(f"[ERROR] {img_file} for {person_name}: {e}")
        
        if person_embeddings:
            embeddings[person_name] = person_embeddings
            
    print(f"[INFO] Built embeddings for {len(embeddings)} people")
    return embeddings

# ----------------- Functions -----------------
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def recognize_face(frame):
    """Recognize face in the given frame"""
    try:
        # Get embedding for the frame
        result = DeepFace.represent(
            frame, 
            model_name="Facenet",
            enforce_detection=False
        )
        
        if not result:
            return "Unknown"
            
        frame_embedding = result[0]["embedding"]
        
        # Compare with all people in the database
        max_similarity = -1  # For cosine similarity
        identity = "Unknown"
        
        for person_name, person_embeddings in embeddings.items():
            # Calculate similarities for all images of this person
            similarities = []
            for db_embedding in person_embeddings:
                similarity = cosine_similarity(db_embedding, frame_embedding)
                similarities.append(similarity)
            
            # Take the maximum similarity (best match) for this person
            person_max_similarity = max(similarities)
            
            if person_max_similarity > max_similarity:
                max_similarity = person_max_similarity
                identity = person_name
        
        # Threshold for cosine similarity (0.4-0.6 is usually good)
        if max_similarity > 0.4:  # Adjust this value based on testing
            return f"{identity} ({max_similarity:.3f})"
        else:
            return f"Unknown ({max_similarity:.3f})"
            
    except Exception as e:
        print(f"[ERROR] in recognize_face: {e}")
        return "Error"

# Main function
def main():
    global embeddings
    
    # Build embeddings first
    print("Building embeddings from database...")
    build_embeddings("database")
    
    if not embeddings:
        print("[ERROR] No embeddings found! Make sure your database folder exists and contains images.")
        return
    
    # Initialize webcam - use IP Webcam
    #capture = cv2.VideoCapture("http://192.168.1.22:8080/video")  # Change IP to yours
    capture = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not capture.isOpened():
        print("Error: Could not open camera stream.")
        return
    
    # Set frame size
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting face recognition... Press 'q' to quit")
    
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame if too big
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Mirror the frame
        mirrored_frame = cv2.flip(frame, 1)
        
        # Recognize face
        identity = recognize_face(mirrored_frame)
        
        # Get frame dimensions
        h, w = mirrored_frame.shape[:2]
        
        # Draw bigger rectangle (proportional to frame size)
        rect_width = int(w * 0.6)  # 60% of frame width
        rect_height = int(h * 0.7)  # 70% of frame height
        x1 = int((w - rect_width) / 2)  # Center horizontally
        y1 = int((h - rect_height) / 2)  # Center vertically
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        
        cv2.rectangle(mirrored_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Put text
        cv2.putText(
            mirrored_frame, 
            f"{identity}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 0), 
            2
        )
        
        # Show the frame in reasonable size
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", 640, 480)
        cv2.imshow("Face Recognition", mirrored_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()