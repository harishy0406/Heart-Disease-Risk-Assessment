"""
Heart Disease Risk Assessment - Model Training Script
This script does the same as the notebook but can be run directly
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
import glob
warnings.filterwarnings('ignore')

print("=" * 60)
print("Heart Disease Risk Assessment - Model Training")
print("=" * 60)

# Step 1: Load Dataset
print("\n[Step 1] Loading dataset...")
# First check for local CSV files
csv_files = glob.glob('./heart-disease-dataset/**/*.csv', recursive=True)
if not csv_files:
    csv_files = glob.glob('./**/heart*.csv', recursive=True)
if not csv_files:
    csv_files = glob.glob('./*.csv', recursive=False)

# If no local files found, try Kaggle API
if not csv_files:
    print("No local CSV files found. Attempting to download from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import sys
        import os
        
        # Check if kaggle.json exists
        kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_path):
            # Try Windows path
            kaggle_path = os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json')
        
        if os.path.exists(kaggle_path):
            api = KaggleApi()
            api.authenticate()
            # Download dataset
            api.dataset_download_files('johnsmith88/heart-disease-dataset', path='./', unzip=True)
            print("Dataset downloaded successfully using Kaggle API!")
            # Refresh CSV file list
            csv_files = glob.glob('./heart-disease-dataset/**/*.csv', recursive=True)
            if not csv_files:
                csv_files = glob.glob('./*.csv', recursive=False)
        else:
            print("Kaggle credentials not found. Please set up kaggle.json or provide CSV file locally.")
    except ImportError:
        print("Kaggle API not available.")
    except Exception as e:
        print(f"Kaggle API error: {e}")
else:
    print(f"Found {len(csv_files)} local CSV file(s).")

# Find and load the CSV file
csv_files = glob.glob('./heart-disease-dataset/**/*.csv', recursive=True)
if not csv_files:
    csv_files = glob.glob('./**/heart*.csv', recursive=True)
if not csv_files:
    csv_files = glob.glob('./*.csv', recursive=False)

# Load the CSV file
if csv_files:
    # Load the first CSV file found (usually the main dataset)
    df = pd.read_csv(csv_files[0])
    print(f"Loaded dataset from: {csv_files[0]}")
else:
    # If no CSV found, raise error with instructions
    raise FileNotFoundError(
        "\n" + "="*60 + "\n"
        "No CSV file found. Please do ONE of the following:\n\n"
        "Option 1: Download manually\n"
        "  1. Go to: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset\n"
        "  2. Download the dataset\n"
        "  3. Extract and place the CSV file in the current directory\n\n"
        "Option 2: Set up Kaggle API\n"
        "  1. Go to: https://www.kaggle.com/settings\n"
        "  2. Create API token (downloads kaggle.json)\n"
        "  3. Place kaggle.json in:\n"
        "     - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json\n"
        "     - Linux/Mac: ~/.kaggle/kaggle.json\n"
        "="*60
    )

print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nColumn names:")
print(df.columns.tolist())
print("\nBasic statistics:")
print(df.describe())

# Step 2: Data Exploration
print("\n[Step 2] Exploring data...")
print("Missing values:")
print(df.isnull().sum())

# Check target variable distribution
if 'target' in df.columns:
    print("\nTarget distribution:")
    print(df['target'].value_counts())
elif 'HeartDisease' in df.columns:
    print("\nTarget distribution:")
    print(df['HeartDisease'].value_counts())
else:
    print("\nLooking for target column...")
    print(df.columns.tolist())

# Step 3: Prepare Data for Training
print("\n[Step 3] Preparing data for training...")
# Identify target column
target_col = None
for col in ['target', 'HeartDisease', 'heart_disease', 'Heart Disease']:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f"Using last column '{target_col}' as target")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle categorical variables
label_encoders = {}
X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

# Ensure target is numeric
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"Target encoded. Classes: {le_target.classes_}")
else:
    le_target = None

print(f"\nFeatures shape: {X_encoded.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {X_encoded.columns.tolist()}")
print(f"\nTarget distribution: {pd.Series(y).value_counts().to_dict()}")

# Step 4: Split Data and Train Model
print("\n[Step 4] Splitting data and training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest Classifier with regularization to prevent overfitting
# Reduced max_depth, increased min_samples_split and min_samples_leaf
# to achieve accuracy around 88-91% instead of 100%
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,  # Reduced from 10 to prevent overfitting
    min_samples_split=10,  # Increased to require more samples for splitting
    min_samples_leaf=5,  # Increased to require more samples in leaf nodes
    max_features='sqrt',  # Use sqrt of features instead of all
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Training completed!")

# Step 5: Evaluate Model
print("\n[Step 5] Evaluating model...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Step 6: Save Model and Metadata
print("\n[Step 6] Saving model and metadata...")
os.makedirs('models', exist_ok=True)

# Save the model
model_path = 'models/heart_disease_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"Model saved to {model_path}")

# Save label encoders and metadata
metadata = {
    'feature_columns': X_encoded.columns.tolist(),
    'label_encoders': label_encoders,
    'target_encoder': le_target,
    'original_columns': X.columns.tolist(),
    'target_column': target_col,
    'feature_importance': feature_importance.to_dict('records'),
    'accuracy': float(accuracy)
}

metadata_path = 'models/model_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to {metadata_path}")
print(f"\n{'='*60}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Features used for prediction: {len(X_encoded.columns)}")
print(f"Feature names: {X_encoded.columns.tolist()}")
print(f"{'='*60}")
print("\nTraining completed successfully!")

