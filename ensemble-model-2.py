import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

# --- Deep Learning Imports for Feature Extraction ---
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# --- Scikit-learn & Imblearn Imports for the Ensemble ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- Import All Classifiers for the Ensemble ---
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURATION ---
# Using the dataset that gave the best performance
IMAGE_FOLDER_PATH = "E:/PROGRAMMING-2/Ensemble-Model-Wate-Classification-Policy-Prediction/dataset/Dataset-OR"
RANDOM_STATE = 50
IMG_SIZE = (224, 224)
CLASS_NAMES = ['organic', 'recyclable']

# --- 2. LOAD MobileNetV2 AND EXTRACT DEEP FEATURES ---
print("Step 2: Loading MobileNetV2 model and extracting deep features...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

print("MobileNetV2 model loaded successfully.")

all_features = []
all_labels = []

PREFIX_TO_LABEL_MAP = {
    'o': 0, 'r': 1
}

image_files_to_process = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in tqdm(image_files_to_process, desc="Extracting Features"):
    match = re.match(r"([a-zA-Z]+)", filename)
    if not match: continue
    prefix = match.group(1).lower()
    if prefix in PREFIX_TO_LABEL_MAP:
        label = PREFIX_TO_LABEL_MAP[prefix]
        file_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        try:
            img = load_img(file_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            features = feature_extractor.predict(img_preprocessed)
            all_features.append(features.flatten())
            all_labels.append(label)
        except Exception as e:
            print(f"  - Warning: Could not process image {filename}. Error: {e}")

X = np.array(all_features)
y = np.array(all_labels)
print(f"\nData loaded successfully. Total samples: {len(X)}")

# --- 3. DATA SPLITTING ---
print("\nStep 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# --- 4. BUILD & TRAIN THE FINAL ENSEMBLE MODEL ---
print("\nStep 4: Defining and training the final ENSEMBLE model...")

# Define the individual models for the ensemble
clf1 = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
# Logistic Regression is a fast and effective choice to add diversity
clf3 = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

# Create the VotingClassifier ensemble
# It combines the three classifiers using a majority vote ('hard' voting)
ensemble_model = VotingClassifier(
    estimators=[('lgbm', clf1), ('rf', clf2), ('lr', clf3)],
    voting='hard',
    n_jobs=-1
)

# Create the full pipeline with all steps
model_pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('classifier', ensemble_model)
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