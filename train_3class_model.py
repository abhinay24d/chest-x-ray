import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder_path, label):
    X_local = []
    y_local = []
    print(f"Loading images from {folder_path}...")
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return X_local, y_local
    
    files = os.listdir(folder_path)
    count = 0
    for filename in files:
        # Support common image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(image_path)
            img = img.convert("L")  # Grayscale
            img = img.resize((64, 64))  # Resize
            img_array = np.array(img)
            
            # Verify the shape is indeed 64x64 (4096 elements)
            if img_array.shape == (64, 64):
                X_local.append(img_array.flatten())
                y_local.append(label)
                count += 1
        except Exception as e:
            # Skip corrupted/unreadable images
            pass
            
    print(f"Loaded {count} images successfully for label {label}.")
    return X_local, y_local

def main():
    X = []
    y = []
    
    # 1. Load NORMAL (label 0)
    folders_normal = [
        r"datasets/train/NORMAL",
        r"datasets/test/NORMAL",
        r"Dataset of Tuberculosis Chest X-rays Images/Normal Chest X-rays"
    ]
    for folder in folders_normal:
        x_n, y_n = load_images_from_folder(folder, 0)
        X.extend(x_n)
        y.extend(y_n)
        
    # 2. Load PNEUMONIA (label 1)
    folders_pneumonia = [
        r"datasets/train/PNEUMONIA",
        r"datasets/test/PNEUMONIA"
    ]
    for folder in folders_pneumonia:
        x_p, y_p = load_images_from_folder(folder, 1)
        X.extend(x_p)
        y.extend(y_p)
        
    # 3. Load TUBERCULOSIS (label 2)
    folders_tb = [
        r"Dataset of Tuberculosis Chest X-rays Images/TB Chest X-rays"
    ]
    for folder in folders_tb:
        x_t, y_t = load_images_from_folder(folder, 2)
        X.extend(x_t)
        y.extend(y_t)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"Class distribution: NORMAL = {np.sum(y == 0)}, PNEUMONIA = {np.sum(y == 1)}, TUBERCULOSIS = {np.sum(y == 2)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest Classifier...")
    # Initialize Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]))
    
    # Save the model
    output_dir = "AI_Chest_Xray_WebApp/model"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tb_pneumonia_random_forest.pkl")
    
    print(f"Saving model to {output_path}...")
    joblib.dump(rf_model, output_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
