import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)
            print("Model loaded successfully")
            return True
        return False
    
    def extract_features(self, image_path):
        """Extract HOG features from an image"""
        try:
            # Read and resize the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            image = cv2.resize(image, (128, 128))
            
            # Apply HOG feature extraction
            fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)
            
            # Normalize the features
            fd = fd / np.max(fd)
            
            return fd
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def train(self, normal_dir, infected_dir):
        """Train the classifier"""
        normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        infected_images = [os.path.join(infected_dir, f) for f in os.listdir(infected_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(normal_images)} normal images")
        print(f"Found {len(infected_images)} infected images")
        
        # Extract features and create labels
        features = []
        labels = []
        
        for img_path in normal_images:
            feat = self.extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(0)  # 0 for normal
        
        for img_path in infected_images:
            feat = self.extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(1)  # 1 for infected
        
        if not features:
            raise ValueError("No valid images found for training")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train SVM classifier
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save the model
        dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def predict(self, image_path):
        """Predict whether an image is normal or infected"""
        if not self.model:
            raise ValueError("Model not trained or loaded")
        
        features = self.extract_features(image_path)
        if features is None:
            return "Error: Could not process image"
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        if prediction == 0:
            return f"Normal (confidence: {probability[0]*100:.2f}%)"
        else:
            return f"Infected (confidence: {probability[1]*100:.2f}%)"

def main():
    # Paths to your training folders
    normal_dir = "train/normal"
    infected_dir = "train/infected"
    
    classifier = ImageClassifier()
    
    # Train if model doesn't exist
    if not classifier.load_model():
        print("Training new model...")
        classifier.train(normal_dir, infected_dir)
    
    # Interactive prediction
    while True:
        print("\nEnter the path of an image to classify (or 'quit' to exit):")
        image_path = input().strip()
        
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            print("File not found. Please try again.")
            continue
            
        result = classifier.predict(image_path)
        print(f"\nClassification result: {result}")

if __name__ == "__main__":
    main()