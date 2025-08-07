
"""
Prediction Script for Titanic Survival Prediction
CODSOFT Data Science Internship - Task 1

This script loads trained models and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import os
from data_preprocessing import TitanicDataPreprocessor

warnings.filterwarnings('ignore')

class TitanicPredictor:
    """
    Prediction pipeline for Titanic survival prediction
    """

    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_name = None

    def load_model(self, model_path=None):
        """
        Load a trained model

        Args:
            model_path (str, optional): Path to model file. If None, loads latest best model.
        """

        if model_path is None:
            # Find the latest best model
            model_files = [f for f in os.listdir('models') if f.startswith('best_model_') and f.endswith('.joblib')]

            if not model_files:
                # If no best model, try to find any model
                model_files = [f for f in os.listdir('models') if f.endswith('.joblib') and 'metadata' not in f]

                if not model_files:
                    raise FileNotFoundError("No trained models found. Please run train.py first.")

                # Use the most recent model
                model_files.sort(reverse=True)
                model_path = os.path.join('models', model_files[0])
                self.model_name = model_files[0].replace('.joblib', '')
            else:
                # Use the most recent best model
                model_files.sort(reverse=True)
                model_path = os.path.join('models', model_files[0])
                self.model_name = "Best Model"

        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path}")
            print(f"🏷️ Model type: {type(self.model).__name__}")

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def load_preprocessor(self):
        """Initialize and setup the preprocessor"""
        self.preprocessor = TitanicDataPreprocessor()
        print("✅ Preprocessor initialized")

    def predict_single_passenger(self, passenger_data):
        """
        Predict survival for a single passenger

        Args:
            passenger_data (dict): Dictionary containing passenger information

        Returns:
            dict: Prediction results
        """

        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        if self.preprocessor is None:
            self.load_preprocessor()

        print("\n🔮 Making Survival Prediction...")
        print("="*30)

        # Convert to DataFrame
        df = pd.DataFrame([passenger_data])

        # Add dummy target column for preprocessing
        df['Survived'] = 0  # Will be removed during preprocessing

        try:
            # Preprocess the data
            X, _ = self.preprocessor.preprocess_pipeline(df, 'Survived')

            # Ensure we have the right features
            # Note: In a production system, you'd save feature names during training
            expected_features = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None

            if expected_features is not None:
                # Ensure X has all expected features
                for feature in expected_features:
                    if feature not in X.columns:
                        X[feature] = 0

                # Select only the features used in training
                X = X[expected_features]

            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

            survival_probability = probability[1]
            survival_prediction = int(prediction)

            # Prepare results
            results = {
                'passenger_data': passenger_data,
                'survival_prediction': survival_prediction,
                'survival_probability': survival_probability,
                'predicted_outcome': 'SURVIVED' if survival_prediction == 1 else 'DID NOT SURVIVE',
                'confidence': max(probability),
                'model_used': self.model_name
            }

            # Display results
            print(f"👤 Passenger Profile:")
            for key, value in passenger_data.items():
                print(f"   {key}: {value}")

            print(f"\n🎯 Prediction Results:")
            print(f"   Outcome: {results['predicted_outcome']}")
            print(f"   Survival Probability: {survival_probability:.3f} ({survival_probability*100:.1f}%)")
            print(f"   Confidence: {results['confidence']:.3f}")
            print(f"   Model Used: {self.model_name}")

            return results

        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None

    def predict_batch(self, passengers_file):
        """
        Predict survival for multiple passengers from CSV file

        Args:
            passengers_file (str): Path to CSV file containing passenger data

        Returns:
            pd.DataFrame: DataFrame with predictions
        """

        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        if self.preprocessor is None:
            self.load_preprocessor()

        print(f"\n📊 Making Batch Predictions from {passengers_file}...")
        print("="*50)

        try:
            # Load passenger data
            df = pd.read_csv(passengers_file)
            print(f"📂 Loaded {len(df)} passengers")

            # Add dummy target column if not present
            if 'Survived' not in df.columns:
                df['Survived'] = 0

            # Preprocess the data
            X, _ = self.preprocessor.preprocess_pipeline(df.copy(), 'Survived')

            # Ensure we have the right features
            expected_features = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None

            if expected_features is not None:
                # Ensure X has all expected features
                for feature in expected_features:
                    if feature not in X.columns:
                        X[feature] = 0

                # Select only the features used in training
                X = X[expected_features]

            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            # Create results DataFrame
            results_df = df.copy()
            results_df['Predicted_Survival'] = predictions
            results_df['Survival_Probability'] = probabilities[:, 1]
            results_df['Predicted_Outcome'] = results_df['Predicted_Survival'].map({
                0: 'DID NOT SURVIVE',
                1: 'SURVIVED'
            })
            results_df['Confidence'] = np.max(probabilities, axis=1)

            # Display summary
            survived_count = sum(predictions)
            survival_rate = survived_count / len(predictions)

            print(f"\n📊 Batch Prediction Summary:")
            print(f"   Total Passengers: {len(predictions)}")
            print(f"   Predicted Survivors: {survived_count}")
            print(f"   Predicted Non-Survivors: {len(predictions) - survived_count}")
            print(f"   Predicted Survival Rate: {survival_rate:.2%}")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/reports/batch_predictions_{timestamp}.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")

            return results_df

        except Exception as e:
            print(f"❌ Error in batch prediction: {str(e)}")
            return None

    def get_prediction_explanation(self, passenger_data):
        """
        Get explanation for prediction (for tree-based models)

        Args:
            passenger_data (dict): Passenger information

        Returns:
            dict: Feature importance explanation
        """

        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️ Model doesn't support feature importance explanation")
            return None

        print("\n🔍 Prediction Explanation...")
        print("="*30)

        # Get feature importances
        feature_importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else [f'feature_{i}' for i in range(len(feature_importances))]

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        print("🏆 Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

        return importance_df

def create_sample_passenger_data():
    """Create sample passenger data for testing"""

    sample_passengers = [
        {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 29,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 211.34,
            'Embarked': 'S'
        },
        {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 22,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 7.25,
            'Embarked': 'S'
        },
        {
            'Pclass': 2,
            'Sex': 'female',
            'Age': 35,
            'SibSp': 1,
            'Parch': 2,
            'Fare': 26.0,
            'Embarked': 'C'
        }
    ]

    return sample_passengers

def main():
    """Main prediction function"""

    print("🔮 TITANIC SURVIVAL PREDICTION - PREDICTION SCRIPT")
    print("="*55)

    try:
        # Initialize predictor
        predictor = TitanicPredictor()

        # Load model
        print("\n🤖 Loading trained model...")
        predictor.load_model()

        # Demo with sample passengers
        print("\n🧪 Testing with sample passengers...")
        sample_passengers = create_sample_passenger_data()

        for i, passenger in enumerate(sample_passengers, 1):
            print(f"\n--- Sample Passenger {i} ---")
            results = predictor.predict_single_passenger(passenger)

            if results:
                # Get explanation for tree-based models
                predictor.get_prediction_explanation(passenger)

        print("\n🎉 PREDICTION DEMO COMPLETED!")
        print("="*35)
        print("💡 To use this script:")
        print("   1. Modify passenger_data dictionary with your values")
        print("   2. Call predictor.predict_single_passenger(passenger_data)")
        print("   3. For batch predictions, use predictor.predict_batch('file.csv')")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
