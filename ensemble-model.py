import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
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
RANDOM_STATE = 42
TARGET_COLUMN = 'Flowable'
FEATURE_COLUMNS = ['SectorProducedBy', 'SectorConsumedBy', 'FlowAmount']
# We will keep the top 5 most frequent classes and group the rest.
N_TOP_CLASSES = 5

# --- 2. DATA LOADING ---
print("Step 2: Loading dataset...")
data_file = Path(DATA_PATH)
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

df = pd.read_csv(data_file)
df_model = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
df_model.dropna(subset=[TARGET_COLUMN], inplace=True)
print(f"Loaded and cleaned data: {df_model.shape[0]} rows.")

# --- 3. ROBUST CLASS GROUPING ---
print("\nStep 3: Grouping rare classes by keeping the top N...")
# Get the counts of each material type
class_counts = df_model[TARGET_COLUMN].value_counts()
# Identify the top N most frequent classes
top_classes = class_counts.nlargest(N_TOP_CLASSES).index
print(f"Keeping the top {N_TOP_CLASSES} classes: {top_classes.tolist()}")

# Group all other classes into a single 'Other' category.
# np.where(condition, value_if_true, value_if_false)
df_model[TARGET_COLUMN] = np.where(df_model[TARGET_COLUMN].isin(top_classes), df_model[TARGET_COLUMN], 'Other')
print(f"Number of classes after grouping: {df_model[TARGET_COLUMN].nunique()}")
print("New class distribution:")
print(df_model[TARGET_COLUMN].value_counts())

# --- 4. TARGET VARIABLE PREPARATION ---
print("\nStep 4: Encoding target variable (y)...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_model[TARGET_COLUMN])

# --- 5. FEATURE PREPARATION AND DATA SPLITTING ---
print("\nStep 5: Preparing features (X) and splitting data...")
X = df_model[FEATURE_COLUMNS]
# This split will now work without errors.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")


# --- 6. PREPROCESSING PIPELINE DEFINITION ---
print("\nStep 6: Building feature preprocessing pipeline...")
categorical_features = ['SectorProducedBy', 'SectorConsumedBy']
numeric_features = ['FlowAmount']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 7. ENSEMBLE MODEL DEFINITION ---
print("\nStep 7: Defining the ensemble model...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)), # Reduced n_estimators for small data
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)) # Reduced n_estimators
]
if XGBOOST_AVAILABLE:
    estimators.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=50, random_state=RANDOM_STATE)))

stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3) # Reduced CV folds to 3

# --- 8. MODEL TRAINING ---
print("\nStep 8: Training the Stacking Ensemble Model...")
# Adjusting k_neighbors for SMOTE. It must be less than the smallest class size in the training set.
# We calculate the smallest class size and set k_neighbors accordingly.
min_class_size = pd.Series(y_train).value_counts().min()
smote_k_neighbors = max(1, min_class_size - 1)
print(f"Smallest class in training set has {min_class_size} samples. Setting SMOTE k_neighbors to {smote_k_neighbors}.")

model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors)),
    ('classifier', stacking_classifier)
])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 9. RESULTS AND EVALUATION ---
print("\n--- MODEL EVALUATION RESULTS ---")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix :\n")
print(confusion_matrix(y_test, y_pred))
print("--- END OF SCRIPT ---")