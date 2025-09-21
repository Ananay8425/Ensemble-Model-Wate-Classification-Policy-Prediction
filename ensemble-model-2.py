import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path

# Scikit-learn Imports for Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- 1. CONFIGURATION ---
DATA_PATH = 'E:/PROGRAMMING-2/Ensemble-Model-Wate-Classification-Policy-Prediction/dataset/city_level_data_0_0.csv'
RANDOM_STATE = 42

# --- 2. DATA LOADING ---
print("Step 2: Loading and verifying the dataset...")
data_file = Path(DATA_PATH)
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")
df = pd.read_csv(data_file)
print(f"Loaded data with {df.shape[0]} rows.")


# --- 3. FEATURE AND TARGET SELECTION (with more features) ---
print("\nStep 3: Selecting available features and targets...")
# We are adding more features to see if we can improve the model
potential_feature_cols = [
    'income_id',
    'population_population_number_of_people', # Added Population
    'total_msw_total_msw_generated_tons_year', # Added Total Waste
    'waste_treatment_anaerobic_digestion_percent',
    'waste_treatment_compost_percent',
    'waste_treatment_incineration_percent',
    'waste_treatment_landfill_percent',
    'waste_treatment_recycling_percent'
]
potential_target_cols = [
    'composition_food_organic_waste_percent', 'composition_glass_percent', 'composition_metal_percent',
    'composition_other_percent', 'composition_paper_cardboard_percent', 'composition_plastic_percent',
    'composition_rubber_leather_percent', 'composition_wood_percent', 'composition_yard_garden_green_waste_percent'
]

feature_cols = [col for col in potential_feature_cols if col in df.columns]
target_cols = [col for col in potential_target_cols if col in df.columns]

print(f"\nFound and using {len(feature_cols)} feature columns: {feature_cols}")
print(f"Found and using {len(target_cols)} target columns to predict.")

df_model = df[feature_cols + target_cols].copy()
df_model.dropna(subset=target_cols, how='all', inplace=True)
print(f"Data shape after cleaning: {df_model.shape}")

# The rest of the script is the same...

# --- 4. FEATURE AND TARGET PREPARATION ---
print("\nStep 4: Preparing features (X) and targets (y)...")
X = df_model[feature_cols]
y = df_model[target_cols]
target_imputer = SimpleImputer(strategy='mean')
y = pd.DataFrame(target_imputer.fit_transform(y), columns=y.columns)

# --- 5. DATA SPLITTING ---
print("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

# --- 6. PREPROCESSING PIPELINE DEFINITION ---
print("\nStep 6: Building the feature preprocessing pipeline...")
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

# --- 7. ENSEMBLE REGRESSION MODEL DEFINITION ---
print("\nStep 7: Defining the ensemble regression model...")
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))
]
if XGBOOST_AVAILABLE:
    estimators.append(('xgb', XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)))

stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5)

# --- 8. MODEL TRAINING ---
print("\nStep 8: Training the Stacking Ensemble Regressor...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(stacking_regressor))
])
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 9. RESULTS AND EVALUATION ---
print("\n--- MODEL EVALUATION RESULTS ---")
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = 1 - mse / np.var(y_test)
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Accuracy: {accuracy}")
print("\n--- END OF SCRIPT ---")