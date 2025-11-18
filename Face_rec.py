# Face_rec.py

"""This script performs face recognition verification between two images provided by the user.
 It uses the DeepFace library to extract facial embeddings and compare them."""

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from deepface import DeepFace
import warnings
import os

warnings.filterwarnings("ignore")

# ----------------- Functions -----------------

def load_and_prepare_image(path: str):
    """Load image from path and convert to RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}") # Error handling for missing files
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_user_input():
    """Get two image paths from user input with validation"""
    print("Face Recognition Verification System")
    print("=" * 40)
    
    while True:
        try:
            # Get first image path
            img1_path = input("Enter path to first image: ").strip()
            if not os.path.exists(img1_path):
                print(f"❌ Error: File '{img1_path}' not found. Please try again.")
                continue
            
            # Get second image path
            img2_path = input("Enter path to second image: ").strip()
            if not os.path.exists(img2_path):
                print(f"❌ Error: File '{img2_path}' not found. Please try again.")
                continue
            
            # Validate file extensions
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            if not any(img1_path.lower().endswith(ext) for ext in valid_extensions):
                print(f"❌ Error: '{img1_path}' is not a valid image format. Supported formats: {', '.join(valid_extensions)}")
                continue
                
            if not any(img2_path.lower().endswith(ext) for ext in valid_extensions):
                print(f"❌ Error: '{img2_path}' is not a valid image format. Supported formats: {', '.join(valid_extensions)}")
                continue
            
            return img1_path, img2_path
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit()
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")




def show_images(images, titles=None):
    """Display multiple images side by side"""

    # Display images by matplotlib 
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images, start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)
        plt.axis('off')
        if titles:
            plt.title(titles[i-1])
    plt.show()


def verify_faces(img1, img2, model_name="VGG-Face"):
    """
    Verify if two faces belong to the same person
    and return distance, threshold, and decision
    """
    result = DeepFace.verify(img1, img2, model_name=model_name) # Using DeepFace to verify faces

    # Metrics to measure similarity
    distance = result["distance"]
    threshold = result["threshold"]  
    verified = result["verified"]

    return distance, threshold, verified



# ----------------- Main -----------------

def main():
    try:
        # Get image paths from user
        img1_path, img2_path = get_user_input()
        
        print(f"\n📸 Loading images...")
        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        
        # Load images
        img1 = load_and_prepare_image(img1_path)
        img2 = load_and_prepare_image(img2_path)
        
        # Extract filenames for display
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        
        # Show images
        print(f"\n🖼️  Displaying images...")
        show_images([img1, img2], titles=[f"Image 1: {img1_name}", f"Image 2: {img2_name}"])
        
        # Verify faces
        print(f"\n🔍 Analyzing faces...")
        dist, thr, same = verify_faces(img1, img2)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"Distance: {dist:.4f} (Threshold: {thr:.4f})")
        print(f"Are these the same person? {'Yes ✅' if same else 'No ❌'}")
        
        # Additional analysis
        if same:
            confidence = max(0, (1 - dist/thr) * 100)
            print(f"Confidence: {confidence:.1f}%")
        else:
            print("The faces appear to be from different people.")
            
    except Exception as e:
        print(f"❌ Error during face verification: {e}")
        print("Please make sure the images contain clear faces and try again.")


if __name__ == "__main__":
    main()

