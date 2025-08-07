"""
Configuration settings for Titanic Survival Prediction Project
CODSOFT Data Science Internship - Task 1
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Dataset Configuration
DATASET_CONFIG = {
    "train_file": "Titanic-Dataset.csv",
    "target_column": "Survived",
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "drop_columns": ["PassengerId", "Name", "Ticket", "Cabin"],
    "categorical_columns": ["Sex", "Embarked", "Title"],
    "numerical_columns": ["Age", "Fare", "SibSp", "Parch"],
    "new_features": ["FamilySize", "IsAlone", "Title", "AgeGroup", "FareGroup"],
    "age_bins": 5,
    "fare_bins": 4
}

# Model Configuration
MODEL_CONFIG = {
    "models_to_train": [
        "LogisticRegression",
        "RandomForest", 
        "SVM",
        "GradientBoosting"
    ],
    "cross_validation_folds": 5,
    "scoring_metric": "accuracy",
    "hyperparameter_tuning": False
}

# Hyperparameter Grids (for GridSearchCV)
HYPERPARAMETER_GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

# Visualization Configuration
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl",
    "save_format": "png"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "titanic_prediction.log"
}

# Environment Variables (use with python-dotenv)
ENV_VARIABLES = {
    "DATA_PATH": str(RAW_DATA_DIR),
    "MODEL_PATH": str(MODELS_DIR),
    "RESULTS_PATH": str(RESULTS_DIR)
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "minimum_accuracy": 0.75,
    "minimum_precision": 0.70,
    "minimum_recall": 0.70,
    "minimum_f1_score": 0.70
}

# File Naming Conventions
NAMING_CONVENTIONS = {
    "model_file_pattern": "{model_name}_{timestamp}.joblib",
    "results_file_pattern": "results_{timestamp}.csv",
    "plot_file_pattern": "{plot_type}_{timestamp}.png",
    "timestamp_format": "%Y%m%d_%H%M%S"
}

# Create directories if they don't exist
def create_project_directories():
    """Create all necessary project directories"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created successfully!")

if __name__ == "__main__":
    create_project_directories()
    print("📁 Project structure initialized!")
    print(f"📊 Project root: {PROJECT_ROOT}")
    print(f"📂 Data directory: {DATA_DIR}")
    print(f"🤖 Models directory: {MODELS_DIR}")
    print(f"📈 Results directory: {RESULTS_DIR}")