import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import re
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog # Import the new feature extractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIGURATION ---
IMAGE_FOLDER_PATH = "E:/PROGRAMMING-2/Ensemble-Model-Wate-Classification-Policy-Prediction/dataset/Dataset-OR"
RANDOM_STATE = 50
IMG_SIZE = (64, 64) # HOG works well with this size
CLASS_NAMES = ['organic', 'recyclable']

# --- 2. LOAD DATA AND EXTRACT HOG FEATURES ---
print(f"Step 2: Loading images and extracting HOG features...")
all_features, all_labels = [], []

PREFIX_TO_LABEL_MAP = {
    'o': 0, 'r': 1
}

for filename in os.listdir(IMAGE_FOLDER_PATH):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    match = re.match(r"([a-zA-Z]+)", filename)
    if not match:
        continue
    
    prefix = match.group(1).lower()
    if prefix in PREFIX_TO_LABEL_MAP:
        label = PREFIX_TO_LABEL_MAP[prefix]
        file_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        try:
            img = imread(file_path, as_gray=True)
            img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
            
            # THE KEY CHANGE IS HERE: Extract HOG features instead of flattening
            hog_features = hog(img_resized, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False)
            
            all_features.append(hog_features)
            all_labels.append(label)
        except Exception as e:
            print(f"  - Warning: Could not load image {filename}. Error: {e}")

X, y = np.array(all_features), np.array(all_labels)
print(f"\nData loaded successfully. Total samples: {len(X)}")
print(f"Number of features per image: {X.shape[1]}") # Note the new feature size

# --- 3. DATA SPLITTING ---
print("\nStep 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# --- 4. BUILD & TRAIN THE FINAL OPTIMIZED MODEL ---
print("\nStep 4: Defining and training the final model with SMOTE...")
print("This may take several minutes...")

model_pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=RANDOM_STATE)), # <-- ADD SMOTE HERE
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        # class_weight='balanced' is less necessary with SMOTE, but doesn't hurt
        class_weight='balanced',
        n_jobs=-1
    ))
])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 5. RESULTS AND EVALUATION ---
print("\n--- FINAL MODEL EVALUATION RESULTS ---")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nOverall Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)
print("\n--- END OF SCRIPT ---")