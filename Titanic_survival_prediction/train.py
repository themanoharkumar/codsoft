
"""
Training Script for Titanic Survival Prediction
CODSOFT Data Science Internship - Task 1

This script trains multiple machine learning models and saves the best one.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from data_preprocessing import preprocess_titanic_data
import os

warnings.filterwarnings('ignore')

class TitanicModelTrainer:
    """
    Model training pipeline for Titanic survival prediction
    """

    def __init__(self, random_state=42):
        """
        Initialize the model trainer

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results/reports', exist_ok=True)

    def initialize_models(self):
        """Initialize all machine learning models"""

        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            )
        }

        print(f"✅ Initialized {len(self.models)} models:")
        for model_name in self.models.keys():
            print(f"   • {model_name}")

    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """

        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("="*40)

        for model_name, model in self.models.items():
            print(f"\n🔄 Training {model_name}...")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

                # Store results
                self.results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

                print(f"✅ {model_name}:")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   AUC Score: {auc_score:.4f}")
                print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue

        # Identify best model
        self._identify_best_model()

    def _identify_best_model(self):
        """Identify the best performing model"""
        if not self.results:
            print("❌ No model results available.")
            return

        # Find model with highest accuracy
        best_accuracy = 0
        for model_name, results in self.results.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                self.best_model_name = model_name
                self.best_model = results['model']

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")

    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for specified model

        Args:
            X_train: Training features
            y_train: Training target
            model_name (str): Name of model to tune
        """

        print(f"\n🔧 HYPERPARAMETER TUNING - {model_name}")
        print("="*40)

        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }

        if model_name not in param_grids:
            print(f"❌ No parameter grid defined for {model_name}")
            return None

        # Initialize model
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_name == 'SVM':
            base_model = SVC(random_state=self.random_state, probability=True)
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=self.random_state)

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        print("🔍 Searching for best parameters...")
        grid_search.fit(X_train, y_train)

        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def generate_detailed_report(self, y_test):
        """
        Generate detailed performance report

        Args:
            y_test: True test labels
        """

        print("\n📊 DETAILED PERFORMANCE REPORT")
        print("="*35)

        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC Score': results['auc_score'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)

        print("\n🏆 Model Performance Summary:")
        print(summary_df.to_string(index=False))

        # Detailed report for best model
        if self.best_model_name:
            print(f"\n📋 Detailed Report - {self.best_model_name}:")
            print("-" * (len(self.best_model_name) + 20))
            print(self.results[self.best_model_name]['classification_report'])

            print("\n🔢 Confusion Matrix:")
            cm = self.results[self.best_model_name]['confusion_matrix']
            print(f"True Negatives:  {cm[0,0]}")
            print(f"False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}")
            print(f"True Positives:  {cm[1,1]}")

        # Save summary to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"results/reports/model_performance_{timestamp}.csv"
        summary_df.to_csv(report_filename, index=False)
        print(f"\n💾 Report saved: {report_filename}")

        return summary_df

    def save_models(self):
        """Save all trained models"""

        print("\n💾 SAVING TRAINED MODELS")
        print("="*25)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for model_name, results in self.results.items():
            # Clean model name for filename
            clean_name = model_name.lower().replace(' ', '_').replace('/', '_')
            filename = f"models/{clean_name}_{timestamp}.joblib"

            # Save model
            joblib.dump(results['model'], filename)
            saved_files.append(filename)
            print(f"✅ Saved: {filename}")

        # Save best model separately
        if self.best_model:
            best_model_filename = f"models/best_model_{timestamp}.joblib"
            joblib.dump(self.best_model, best_model_filename)
            saved_files.append(best_model_filename)
            print(f"🏆 Best model saved: {best_model_filename}")

        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'best_model': self.best_model_name,
            'model_files': saved_files,
            'performance_summary': {
                name: {
                    'accuracy': results['accuracy'],
                    'auc_score': results['auc_score'],
                    'cv_mean': results['cv_mean']
                }
                for name, results in self.results.items()
            }
        }

        metadata_filename = f"models/model_metadata_{timestamp}.joblib"
        joblib.dump(metadata, metadata_filename)
        print(f"📋 Metadata saved: {metadata_filename}")

        return saved_files

def main():
    """Main training function"""

    print("🚀 TITANIC SURVIVAL PREDICTION - MODEL TRAINING")
    print("="*50)

    try:
        # Step 1: Load and preprocess data
        print("\n📂 Loading and preprocessing data...")
        X, y, preprocessor = preprocess_titanic_data('Titanic-Dataset.csv')

        # Step 2: Split data
        print("\n✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Step 3: Initialize trainer
        trainer = TitanicModelTrainer(random_state=42)
        trainer.initialize_models()

        # Step 4: Train models
        trainer.train_models(X_train, X_test, y_train, y_test)

        # Step 5: Hyperparameter tuning (optional)
        print("\n🔧 Performing hyperparameter tuning for Random Forest...")
        trainer.hyperparameter_tuning(X_train, y_train, 'Random Forest')

        # Retrain with tuned model
        tuned_model = trainer.models['Random Forest']
        tuned_model.fit(X_train, y_train)

        # Update results
        y_pred_tuned = tuned_model.predict(X_test)
        y_pred_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]
        accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

        trainer.results['Random Forest (Tuned)'] = {
            'model': tuned_model,
            'accuracy': accuracy_tuned,
            'auc_score': roc_auc_score(y_test, y_pred_proba_tuned),
            'cv_mean': cross_val_score(tuned_model, X_train, y_train, cv=5).mean(),
            'cv_std': cross_val_score(tuned_model, X_train, y_train, cv=5).std(),
            'y_pred': y_pred_tuned,
            'y_pred_proba': y_pred_proba_tuned,
            'classification_report': classification_report(y_test, y_pred_tuned),
            'confusion_matrix': confusion_matrix(y_test, y_pred_tuned)
        }

        print(f"✅ Tuned Random Forest Accuracy: {accuracy_tuned:.4f}")

        # Step 6: Generate detailed report
        trainer.generate_detailed_report(y_test)

        # Step 7: Save models
        trainer.save_models()

        print("\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*40)
        print("📁 Check the following directories:")
        print("   🤖 models/ - Trained model files")
        print("   📊 results/reports/ - Performance reports")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
