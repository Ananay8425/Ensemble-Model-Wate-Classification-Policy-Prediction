import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class WasteDataLoader:
    """Handles loading and initial validation of waste management data"""
    
    def __init__(self):
        self.required_columns = ['country_or_region', 'material_type', 'quantity']
        self.optional_columns = ['year', 'management_type', 'population', 'gdp']
    
    def load_data(self, file_path):
        """Load CSV data with validation"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return self.validate_data(df)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def validate_data(self, df):
        """Validate required columns and data quality"""
        print("\n=== Data Validation ===")
        
        # Check required columns
        missing_required = [col for col in self.required_columns if col not in df.columns]
        if missing_required:
            print(f"ERROR: Missing required columns: {missing_required}")
            return None
        
        # Check optional columns
        missing_optional = [col for col in self.optional_columns if col not in df.columns]
        if missing_optional:
            print(f"Warning: Missing optional columns: {missing_optional}")
        
        # Show available columns
        print(f"Available columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Data shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        return df

class DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def preprocess(self, df):
        """Clean and preprocess the data"""
        print("\n=== Data Preprocessing ===")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Drop rows with missing essential data
        initial_rows = len(df)
        df = df.dropna(subset=['material_type', 'country_or_region'])
        print(f"Dropped {initial_rows - len(df)} rows with missing essential data")
        
        # Convert quantity to numeric (this is essential)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        # Convert year to numeric if it exists, but handle missing values
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            # Fill missing years with a default value (e.g., median year or a placeholder)
            year_median = df['year'].median()
            if pd.isna(year_median):
                df['year'] = df['year'].fillna(2020)  # Default year if all are missing
            else:
                df['year'] = df['year'].fillna(year_median)
        
        # Handle optional columns only if they exist
        if 'population' in df.columns:
            df['population'] = pd.to_numeric(df['population'], errors='coerce')
        
        if 'gdp' in df.columns:
            df['gdp'] = pd.to_numeric(df['gdp'], errors='coerce')
        
        # Remove rows where quantity is still missing (essential for analysis)
        df = df.dropna(subset=['quantity'])
        
        print(f"Final data shape after preprocessing: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        return df

class SMOTEHandler:
    """Handles SMOTE oversampling for class imbalance"""
    
    def __init__(self, sampling_strategy='auto', k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.smote = None
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to training data"""
        print("\n=== Applying SMOTE for Class Imbalance ===")
        
        # Check class distribution before SMOTE
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        print("Class distribution before SMOTE:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples")
        
        # Adjust k_neighbors based on smallest class size
        min_class_size = min(class_counts)
        k_neighbors = min(self.k_neighbors, min_class_size - 1)
        
        if k_neighbors < 1:
            print("Warning: Some classes have too few samples for SMOTE. Using class weights instead.")
            return X_train, y_train, False
        
        try:
            # Initialize SMOTE with adjusted parameters
            self.smote = SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )
            
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
            # Check class distribution after SMOTE
            unique_classes_after, class_counts_after = np.unique(y_resampled, return_counts=True)
            print(f"\nClass distribution after SMOTE:")
            for cls, count in zip(unique_classes_after, class_counts_after):
                print(f"  Class {cls}: {count} samples")
            
            print(f"Data shape before SMOTE: {X_train.shape}")
            print(f"Data shape after SMOTE: {X_resampled.shape}")
            
            return X_resampled, y_resampled, True
            
        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Continuing without SMOTE...")
            return X_train, y_train, False

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and visualizations"""
    
    def __init__(self, material_map):
        self.material_map = material_map
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive evaluation of a single model"""
        print(f"\n=== {model_name} Evaluation ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        print(f"Accuracy Score: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        
        # Classification Report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{model_name} Confusion Matrix:")
        print(cm)
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name="Model", figsize=(12, 10)):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=figsize)
        
        # Create labels from material mapping
        labels = [self.material_map.get(i, f'Class_{i}') for i in range(len(cm))]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, results_dict):
        """Compare multiple models and show the best performer"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1 (Weighted)': results['f1_weighted'],
                'F1 (Macro)': results['f1_macro'],
                'F1 (Micro)': results['f1_micro']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Find best model for each metric
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_f1_weighted = comparison_df.loc[comparison_df['F1 (Weighted)'].idxmax(), 'Model']
        best_f1_macro = comparison_df.loc[comparison_df['F1 (Macro)'].idxmax(), 'Model']
        
        print(f"\n🏆 Best Models:")
        print(f"   Accuracy: {best_accuracy} ({comparison_df['Accuracy'].max():.4f})")
        print(f"   F1 (Weighted): {best_f1_weighted} ({comparison_df['F1 (Weighted)'].max():.4f})")
        print(f"   F1 (Macro): {best_f1_macro} ({comparison_df['F1 (Macro)'].max():.4f})")
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'F1 (Micro)']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Load and preprocess data
loader = WasteDataLoader()
df = loader.load_data('dataset/harmonized_merged_waste_data.csv')

if df is not None:
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df)


    # encode material_type target for classification
    df['material_label'] = df['material_type'].astype('category').cat.codes
    material_map = dict(enumerate(df['material_type'].astype('category').cat.categories))
    print(f"\nMaterial mapping: {material_map}")
    
    # Enhanced feature engineering
    print("\n=== Feature Engineering ===")
    
    # Create derived features
    df['log_quantity'] = np.log1p(df['quantity'])  # Log transform for skewed data
    df['quantity_squared'] = df['quantity'] ** 2   # Non-linear relationship
    
    # Create quantity bins for categorical analysis
    df['quantity_bin'] = pd.cut(df['quantity'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Year-based features if available
    if 'year' in df.columns and not df['year'].isna().all():
        current_year = 2024
        df['years_since'] = current_year - df['year']
        df['decade'] = (df['year'] // 10) * 10
    
    # Choose available feature columns
    available_numeric_cols = ['quantity', 'log_quantity']
    
    if 'year' in df.columns and not df['year'].isna().all():
        available_numeric_cols.extend(['year', 'years_since'])
    
    # Check if quantity_per_1000 exists, if not create it
    if 'quantity_per_1000' not in df.columns and 'population' in df.columns:
        df['quantity_per_1000'] = df['quantity'] / (df['population'] / 1000)
        available_numeric_cols.append('quantity_per_1000')
    
    feature_cols = available_numeric_cols
    cat_cols = ['country_or_region']
    
    # Add management_type if it has useful information
    if 'management_type' in df.columns:
        # Check if management_type has any non-null values
        non_null_mgmt = df['management_type'].notna().sum()
        if non_null_mgmt > 0:
            print(f"Management type has {non_null_mgmt} non-null values, including it as feature")
            cat_cols.append('management_type')
        else:
            print("Management type is all null, excluding from features")
    
    # Add quantity bins as categorical feature
    cat_cols.append('quantity_bin')
    
    print(f"Using numeric features: {feature_cols}")
    print(f"Using categorical features: {cat_cols}")
    
    # Prepare features and target (don't drop rows with missing features yet)
    X = df[feature_cols + cat_cols]
    y = df['material_label']
    
    print(f"Initial data shape: {X.shape}")
    print(f"Material type distribution:")
    material_counts = y.value_counts()
    print(material_counts)
    
    # Analyze class imbalance
    print(f"\nClass imbalance analysis:")
    print(f"Most common class: {material_counts.iloc[0]} samples ({material_counts.iloc[0]/len(y)*100:.1f}%)")
    print(f"Least common class: {material_counts.iloc[-1]} samples ({material_counts.iloc[-1]/len(y)*100:.1f}%)")
    print(f"Imbalance ratio: {material_counts.iloc[0]/material_counts.iloc[-1]:.1f}:1")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(f"Quantity range: {X['quantity'].min():.2f} to {X['quantity'].max():.2f}")
    print(f"Quantity mean: {X['quantity'].mean():.2f}")
    print(f"Number of unique countries: {X['country_or_region'].nunique()}")
    
    if 'year' in X.columns:
        print(f"Year range: {X['year'].min():.0f} to {X['year'].max():.0f}")
    
    # Filter out classes with very few samples (less than 2) to enable stratification
    min_samples_per_class = 2
    valid_classes = material_counts[material_counts >= min_samples_per_class].index
    
    if len(valid_classes) < len(material_counts):
        print(f"\nFiltering out classes with < {min_samples_per_class} samples")
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
        print(f"Data shape after filtering rare classes: {X.shape}")
    
    # Check if we have enough data for train-test split
    if len(X) < 10:
        print("ERROR: Insufficient data for machine learning. Need at least 10 samples.")
        exit()
    
    # Adjust test size based on available data
    n_classes = y.nunique()
    min_test_size = max(n_classes, 5)  # At least one sample per class or 5 samples
    
    if len(X) * 0.2 < min_test_size:
        test_size = min_test_size / len(X)
        test_size = min(test_size, 0.5)  # Don't use more than 50% for testing
        print(f"Adjusting test_size to {test_size:.2f} to ensure sufficient samples per class")
    else:
        test_size = 0.2
    
    # Split data with or without stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        print("Using stratified split")
    except ValueError as e:
        print(f"Stratification failed: {e}")
        print("Using random split without stratification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    print(f"\nData split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Define preprocessing pipelines
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    
    # Apply preprocessing to training data first (needed for SMOTE)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote_handler = SMOTEHandler(sampling_strategy='auto', k_neighbors=5)
    X_train_smote, y_train_smote, smote_applied = smote_handler.apply_smote(X_train_processed, y_train)
    
    if smote_applied:
        print("SMOTE successfully applied!")
        X_train_final = X_train_smote
        y_train_final = y_train_smote
    else:
        print("Using original training data without SMOTE")
        X_train_final = X_train_processed
        y_train_final = y_train
    
    # Create models with appropriate class weighting
    if smote_applied:
        # If SMOTE was applied, we don't need class_weight='balanced'
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        # If SMOTE wasn't applied, use class weighting
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print("\n=== Training Ensemble Model ===")
    print("Training Random Forest and Gradient Boosting ensemble...")
    
    # Train models directly on processed data
    rf_model.fit(X_train_final, y_train_final)
    gb_model.fit(X_train_final, y_train_final)
    
    # Create voting classifier
    vc = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)], 
        voting='soft'
    )
    vc.fit(X_train_final, y_train_final)
    
    print("Training individual models for comparison...")
    
    # Create separate models for comparison
    rf_comparison = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced' if not smote_applied else None,
        random_state=42
    )
    
    gb_comparison = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_comparison.fit(X_train_final, y_train_final)
    gb_comparison.fit(X_train_final, y_train_final)


    # Initialize comprehensive evaluator
    evaluator = ModelEvaluator(material_map)
    
    # Evaluate all models comprehensively
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Show SMOTE information
    if smote_applied:
        print(f"📈 SMOTE Applied: Training samples increased from {len(y_train)} to {len(y_train_final)}")
        print(f"   Balanced {len(np.unique(y_train))} classes")
    else:
        print("⚠️  SMOTE not applied - using class weights for imbalance handling")
    
    # Evaluate each model
    rf_results = evaluator.evaluate_model(rf_comparison, X_test_processed, y_test, "Random Forest")
    gb_results = evaluator.evaluate_model(gb_comparison, X_test_processed, y_test, "Gradient Boosting")
    ensemble_results = evaluator.evaluate_model(vc, X_test_processed, y_test, "Ensemble (Voting)")
    
    # Compare all models
    all_results = {
        'Random Forest': rf_results,
        'Gradient Boosting': gb_results,
        'Ensemble': ensemble_results
    }
    
    comparison_df = evaluator.compare_models(all_results)
    
    # Plot model comparison
    evaluator.plot_model_comparison(comparison_df)
    
    # Plot confusion matrices for best model
    best_model_name = comparison_df.loc[comparison_df['F1 (Weighted)'].idxmax(), 'Model']
    best_results = all_results[best_model_name]
    
    print(f"\n=== Detailed Analysis for Best Model: {best_model_name} ===")
    evaluator.plot_confusion_matrix(best_results['confusion_matrix'], best_model_name)
    
    # Show top predicted classes
    unique_preds, pred_counts = np.unique(y_pred_ensemble, return_counts=True)
    print(f"\nTop predicted material types:")
    for pred_class, count in sorted(zip(unique_preds, pred_counts), key=lambda x: x[1], reverse=True)[:5]:
        material_name = material_map[pred_class]
        print(f"  {material_name}: {count} predictions")
    
    # Feature importance visualization
    try:
        # Get feature importance from Random Forest
        feature_importance = rf_comparison.feature_importances_
        
        # Get feature names after preprocessing
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
            feature_names.extend(cat_feature_names)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance (top 20 features for readability)
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importance (Random Forest with SMOTE)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")
    
    print(f"\nMaterial type distribution in test set:")
    test_materials = [material_map[i] for i in y_test]
    print(pd.Series(test_materials).value_counts())

else:
    print("Failed to load data. Please check the file path and data format.")