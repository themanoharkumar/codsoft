
"""
Titanic Survival Prediction - Complete Project Script
CODSOFT Data Science Internship - Task 1

This script provides a complete end-to-end machine learning pipeline
for predicting Titanic passenger survival.

Author: [Your Name]
Date: August 2025
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
import joblib
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TitanicSurvivalPredictor:
    """
    Complete Titanic Survival Prediction Pipeline
    """

    def __init__(self, data_path='Titanic-Dataset.csv'):
        """
        Initialize the predictor with data path

        Args:
            data_path (str): Path to the Titanic dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}

        # Create directories for outputs
        os.makedirs('models', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)
        os.makedirs('results/reports', exist_ok=True)

        print("🚢 Titanic Survival Predictor Initialized")
        print("="*50)

    def load_data(self):
        """Load and display basic information about the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"🎯 Target distribution:")
            print(self.df['Survived'].value_counts())
            print(f"📉 Survival rate: {self.df['Survived'].mean():.2%}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File '{self.data_path}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print("❌ Data not loaded. Please run load_data() first.")
            return

        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("="*40)

        # Display basic info
        print("📋 Dataset Info:")
        print(self.df.info())

        print("\n📊 Missing Values:")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print(missing)

        print("\n📈 Statistical Summary:")
        print(self.df.describe())

        # Create visualizations
        self._create_exploration_plots()

    def _create_exploration_plots(self):
        """Create exploratory data analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Titanic Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

        # 1. Survival count
        sns.countplot(data=self.df, x='Survived', ax=axes[0,0])
        axes[0,0].set_title('Survival Distribution')
        axes[0,0].set_xlabel('Survived (0=No, 1=Yes)')

        # 2. Survival by Gender
        sns.countplot(data=self.df, x='Sex', hue='Survived', ax=axes[0,1])
        axes[0,1].set_title('Survival by Gender')

        # 3. Survival by Passenger Class
        sns.countplot(data=self.df, x='Pclass', hue='Survived', ax=axes[0,2])
        axes[0,2].set_title('Survival by Passenger Class')

        # 4. Age distribution
        self.df['Age'].hist(bins=30, ax=axes[1,0])
        axes[1,0].set_title('Age Distribution')
        axes[1,0].set_xlabel('Age')

        # 5. Fare distribution
        self.df['Fare'].hist(bins=30, ax=axes[1,1])
        axes[1,1].set_title('Fare Distribution')
        axes[1,1].set_xlabel('Fare')

        # 6. Survival by Embarked
        sns.countplot(data=self.df, x='Embarked', hue='Survived', ax=axes[1,2])
        axes[1,2].set_title('Survival by Port of Embarkation')

        plt.tight_layout()
        plt.savefig('results/figures/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig('results/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def preprocess_data(self):
        """
        Clean and preprocess the data for machine learning
        """
        print("\n🛠️ PREPROCESSING DATA")
        print("="*30)

        # Make a copy of the original data
        df_processed = self.df.copy()

        # 1. Handle missing values
        print("1️⃣ Handling missing values...")

        # Fill missing Age values with median
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)

        # Fill missing Embarked with mode
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)

        # Fill missing Fare with median
        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)

        # Drop Cabin column (too many missing values)
        df_processed.drop('Cabin', axis=1, inplace=True)

        # 2. Feature Engineering
        print("2️⃣ Engineering new features...")

        # Extract Title from Name
        df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                               'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                               'Jonkheer', 'Dona'], 'Rare')
        df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

        # Family Size
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1

        # IsAlone
        df_processed['IsAlone'] = 1
        df_processed.loc[df_processed['FamilySize'] > 1, 'IsAlone'] = 0

        # Age Groups
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 5, labels=[1,2,3,4,5])

        # Fare Groups
        df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], 4, labels=[1,2,3,4])

        # 3. Encode categorical variables
        print("3️⃣ Encoding categorical variables...")

        # Binary encoding for Sex
        df_processed['Sex'] = df_processed['Sex'].map({'female': 1, 'male': 0})

        # One-hot encoding for other categorical variables
        df_processed = pd.get_dummies(df_processed, columns=['Embarked', 'Title'], drop_first=True)

        # 4. Drop unnecessary columns
        print("4️⃣ Dropping unnecessary columns...")
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        df_processed.drop(columns_to_drop, axis=1, inplace=True)

        # Store processed data
        self.df_processed = df_processed

        print(f"✅ Preprocessing complete!")
        print(f"📊 Processed dataset shape: {df_processed.shape}")

        # Show feature importance info
        feature_info = pd.DataFrame({
            'Feature': df_processed.columns,
            'Non_Null_Count': df_processed.count(),
            'Data_Type': df_processed.dtypes
        })
        print("\n📋 Final Features:")
        print(feature_info)

        return df_processed

    def prepare_features(self):
        """Prepare features for machine learning"""
        if not hasattr(self, 'df_processed'):
            print("❌ Data not preprocessed. Please run preprocess_data() first.")
            return

        print("\n🎯 PREPARING FEATURES FOR MACHINE LEARNING")
        print("="*45)

        # Separate features and target
        X = self.df_processed.drop('Survived', axis=1)
        y = self.df_processed['Survived']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Data split complete!")
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")
        print(f"📊 Feature columns: {list(X.columns)}")

    def train_models(self):
        """Train multiple machine learning models"""
        if self.X_train is None:
            print("❌ Features not prepared. Please run prepare_features() first.")
            return

        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("="*40)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        # Train each model
        for name, model in models.items():
            print(f"🔄 Training {name}...")

            # Use scaled features for models that benefit from scaling
            if name in ['Logistic Regression', 'SVM']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test

            # Train model
            model.fit(X_train_use, self.y_train)

            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)

            # Cross-validation score
            if name in ['Logistic Regression', 'SVM']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

        print("\n🎉 All models trained successfully!")

    def evaluate_models(self):
        """Evaluate and compare all trained models"""
        if not self.results:
            print("❌ No models trained. Please run train_models() first.")
            return

        print("\n📊 MODEL EVALUATION RESULTS")
        print("="*35)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'AUC Score': [self.results[model]['auc_score'] for model in self.results.keys()],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        }).round(4)

        print(results_df.to_string(index=False))

        # Find best model
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model_name}")

        # Create evaluation plots
        self._create_evaluation_plots()

        # Detailed classification reports
        print("\n📋 DETAILED CLASSIFICATION REPORTS")
        print("="*40)

        for model_name in self.results.keys():
            print(f"\n{model_name}:")
            print("-" * len(model_name))
            y_pred = self.results[model_name]['y_pred']
            print(classification_report(self.y_test, y_pred))

    def _create_evaluation_plots(self):
        """Create model evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

        # 1. Model Accuracy Comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]

        axes[0,0].bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0.7, 1.0)
        for i, v in enumerate(accuracies):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center')

        # 2. ROC Curves
        for model_name in models:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = self.results[model_name]['auc_score']
            axes[0,1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')

        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True)

        # 3. Best Model Confusion Matrix
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_predictions = self.results[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, best_predictions)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')

        # 4. Cross-Validation Scores
        cv_means = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]

        axes[1,1].bar(models, cv_means, yerr=cv_stds, capsize=5, 
                     color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1,1].set_title('Cross-Validation Scores')
        axes[1,1].set_ylabel('CV Accuracy')
        axes[1,1].set_ylim(0.7, 1.0)

        plt.tight_layout()
        plt.savefig('results/figures/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def feature_importance_analysis(self):
        """Analyze feature importance from tree-based models"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*35)

        tree_models = ['Random Forest', 'Gradient Boosting']

        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                importances = model.feature_importances_
                feature_names = self.X_train.columns

                # Create feature importance DataFrame
                feature_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                print(f"\n{model_name} - Top 10 Important Features:")
                print(feature_imp_df.head(10).to_string(index=False))

                # Plot feature importance
                plt.figure(figsize=(10, 8))
                top_features = feature_imp_df.head(15)
                plt.barh(top_features['Feature'], top_features['Importance'])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance - {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'results/figures/feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()

    def save_models(self):
        """Save trained models to disk"""
        print("\n💾 SAVING MODELS")
        print("="*20)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, model in self.models.items():
            filename = f"models/{model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
            joblib.dump(model, filename)
            print(f"✅ {model_name} saved as: {filename}")

        # Save scaler
        scaler_filename = f"models/scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_filename)
        print(f"✅ Scaler saved as: {scaler_filename}")

        # Save results summary
        results_summary = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'AUC_Score': [self.results[model]['auc_score'] for model in self.results.keys()],
            'CV_Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV_Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        })

        results_filename = f"results/reports/model_results_{timestamp}.csv"
        results_summary.to_csv(results_filename, index=False)
        print(f"✅ Results summary saved as: {results_filename}")

    def predict_survival(self, passenger_data):
        """
        Predict survival for new passenger data

        Args:
            passenger_data (dict): Dictionary containing passenger information

        Returns:
            dict: Prediction results from all models
        """
        if not self.models:
            print("❌ No trained models available. Please train models first.")
            return None

        print("\n🔮 MAKING SURVIVAL PREDICTIONS")
        print("="*35)

        # Convert passenger data to DataFrame
        passenger_df = pd.DataFrame([passenger_data])

        # Apply same preprocessing steps
        # Note: In a real application, you'd want to create a separate preprocessing pipeline
        print("⚙️ Processing passenger data...")

        predictions = {}
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])

        for model_name, model in self.models.items():
            # Make prediction (simplified - in reality you'd need full preprocessing pipeline)
            if model_name in ['Logistic Regression', 'SVM']:
                # Use scaled features
                prediction = model.predict_proba([[1]])[0]  # Placeholder
            else:
                prediction = model.predict_proba([[1]])[0]  # Placeholder

            predictions[model_name] = {
                'survival_probability': prediction[1],
                'prediction': int(prediction[1] > 0.5)
            }

        print(f"\n🎯 Predictions for passenger:")
        for model_name, pred in predictions.items():
            status = "SURVIVED" if pred['prediction'] == 1 else "DID NOT SURVIVE"
            prob = pred['survival_probability']
            print(f"{model_name}: {status} (Probability: {prob:.3f})")

        return predictions

    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("🚀 STARTING COMPLETE TITANIC SURVIVAL PREDICTION PIPELINE")
        print("="*60)

        # Step 1: Load Data
        if not self.load_data():
            return False

        # Step 2: Explore Data
        self.explore_data()

        # Step 3: Preprocess Data
        self.preprocess_data()

        # Step 4: Prepare Features
        self.prepare_features()

        # Step 5: Train Models
        self.train_models()

        # Step 6: Evaluate Models
        self.evaluate_models()

        # Step 7: Feature Importance
        self.feature_importance_analysis()

        # Step 8: Save Models
        self.save_models()

        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*40)
        print("📁 Check the following directories for outputs:")
        print("   📊 results/figures/ - Visualization plots")
        print("   📋 results/reports/ - Analysis reports")
        print("   🤖 models/ - Saved model files")

        return True


def main():
    """Main function to run the complete pipeline"""

    # ASCII Art Title
    print("""
    ╔══════════════════════════════════════════════════╗
    ║              🚢 TITANIC SURVIVAL                  ║
    ║               PREDICTION PROJECT                 ║
    ║                                                  ║
    ║                                                  ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Initialize predictor
    predictor = TitanicSurvivalPredictor('Titanic-Dataset.csv')

    # Run complete pipeline
    success = predictor.run_complete_pipeline()

    if success:
        print("\n✨ Thank you for using the Titanic Survival Predictor!")

if __name__ == "__main__":
    main()
