import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- 1. CONFIGURATION ---
DATA_PATH = 'E:/PROGRAMMING-2/Ensemble-Model-Wate-Classification-Policy-Prediction/dataset/Waste_national_2018_v1.3.2_9b1bb41.csv'
RANDOM_STATE = 50
TARGET_COLUMN = 'Flowable'
FEATURE_COLUMNS = ['SectorProducedBy', 'SectorConsumedBy', 'FlowAmount']

# --- 2. DATA LOADING ---
print("Step 2: Loading dataset...")
data_file = Path(DATA_PATH)
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

df = pd.read_csv(data_file)
df_model = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
df_model.dropna(subset=[TARGET_COLUMN], inplace=True)
print(f"Loaded and cleaned data: {df_model.shape[0]} rows.")

# --- 3. TARGET VARIABLE PREPARATION ---
print("\nStep 3: Encoding target variable (y)...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_model[TARGET_COLUMN])

# --- 4. FEATURE PREPARATION AND DATA SPLITTING ---
print("\nStep 4: Preparing features (X) and splitting data...")
X = df_model[FEATURE_COLUMNS]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# --- 5. PREPROCESSING PIPELINE DEFINITION ---
print("\nStep 5: Building feature preprocessing pipeline...")
categorical_features = ['SectorProducedBy', 'SectorConsumedBy']
numeric_features = ['FlowAmount']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 6. ENSEMBLE MODEL DEFINITION ---
print("\nStep 6: Defining the ensemble model...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
]
if XGBOOST_AVAILABLE:
    estimators.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)))

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# --- 7. MODEL TRAINING ---
print("\nStep 7: Training the Stacking Ensemble Model...")
# The final pipeline combines preprocessing, SMOTE for balancing, and the classifier.
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
    ('classifier', stacking_classifier)
])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 8. RESULTS AND EVALUATION ---
print("\n--- MODEL EVALUATION RESULTS ---")
y_pred = model_pipeline.predict(X_test)
print("Predictions : ", y_pred)

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score : ", accuracy_score(y_test, y_pred))