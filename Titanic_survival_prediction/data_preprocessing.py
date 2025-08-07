
"""
Data Preprocessing Module for Titanic Survival Prediction
CODSOFT Data Science Internship - Task 1

This module contains all data cleaning and preprocessing functions.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')

class TitanicDataPreprocessor:
    """
    Data preprocessing pipeline for Titanic dataset
    """

    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate the Titanic dataset

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        try:
            df = pd.read_csv(filepath)

            # Basic validation
            required_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                              'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            print(f"✅ Dataset loaded successfully: {df.shape}")
            print(f"📊 Survival rate: {df['Survived'].mean():.2%}")

            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_clean = df.copy()

        print("🛠️ Handling missing values...")

        # Age: Fill with median grouped by Pclass and Sex
        for pclass in df_clean['Pclass'].unique():
            for sex in df_clean['Sex'].unique():
                mask = (df_clean['Pclass'] == pclass) & (df_clean['Sex'] == sex)
                median_age = df_clean.loc[mask, 'Age'].median()
                df_clean.loc[mask & df_clean['Age'].isna(), 'Age'] = median_age

        # If still missing, fill with overall median
        df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)

        # Embarked: Fill with mode
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

        # Fare: Fill with median grouped by Pclass
        for pclass in df_clean['Pclass'].unique():
            mask = df_clean['Pclass'] == pclass
            median_fare = df_clean.loc[mask, 'Fare'].median()
            df_clean.loc[mask & df_clean['Fare'].isna(), 'Fare'] = median_fare

        # Cabin: Create a binary feature for having cabin info
        df_clean['HasCabin'] = (~df_clean['Cabin'].isna()).astype(int)
        df_clean.drop('Cabin', axis=1, inplace=True)

        print(f"✅ Missing values handled. Remaining: {df_clean.isnull().sum().sum()}")

        return df_clean

    def extract_title_from_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract title from passenger names

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with extracted title feature
        """
        df_titled = df.copy()

        # Extract title
        df_titled['Title'] = df_titled['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

        # Group rare titles
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }

        df_titled['Title'] = df_titled['Title'].map(title_mapping).fillna('Rare')

        print(f"✅ Title extracted. Distribution:")
        print(df_titled['Title'].value_counts())

        return df_titled

    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create family-related features

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with family features
        """
        df_family = df.copy()

        # Family size
        df_family['FamilySize'] = df_family['SibSp'] + df_family['Parch'] + 1

        # Is alone
        df_family['IsAlone'] = (df_family['FamilySize'] == 1).astype(int)

        # Family size categories
        df_family['FamilySize_Category'] = pd.cut(
            df_family['FamilySize'], 
            bins=[0, 1, 4, 11], 
            labels=['Alone', 'Small', 'Large']
        )

        print("✅ Family features created:")
        print(f"   - Family Size range: {df_family['FamilySize'].min()} to {df_family['FamilySize'].max()}")
        print(f"   - Alone passengers: {df_family['IsAlone'].sum()} ({df_family['IsAlone'].mean():.1%})")

        return df_family

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned features for numerical variables

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with binned features
        """
        df_binned = df.copy()

        # Age bins
        df_binned['AgeGroup'] = pd.cut(
            df_binned['Age'], 
            bins=5, 
            labels=['Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
        )

        # Fare bins
        df_binned['FareGroup'] = pd.qcut(
            df_binned['Fare'], 
            q=4, 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )

        print("✅ Binned features created:")
        print(f"   - Age groups: {df_binned['AgeGroup'].value_counts().to_dict()}")
        print(f"   - Fare groups: {df_binned['FareGroup'].value_counts().to_dict()}")

        return df_binned

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = df.copy()

        # Binary encoding for Sex
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

        # One-hot encoding for multi-category variables
        categorical_features = ['Embarked', 'Title', 'AgeGroup', 'FareGroup', 'FamilySize_Category']

        for feature in categorical_features:
            if feature in df_encoded.columns:
                # Get dummies and drop first category to avoid multicollinearity
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(feature, axis=1, inplace=True)

        print(f"✅ Categorical features encoded. New shape: {df_encoded.shape}")

        return df_encoded

    def remove_unnecessary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary features for modeling

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with unnecessary features removed
        """
        df_clean = df.copy()

        # Features to remove
        features_to_drop = ['PassengerId', 'Name', 'Ticket']

        # Only drop features that exist in the dataframe
        features_to_drop = [f for f in features_to_drop if f in df_clean.columns]

        if features_to_drop:
            df_clean.drop(features_to_drop, axis=1, inplace=True)
            print(f"✅ Removed features: {features_to_drop}")

        print(f"📊 Final feature set: {list(df_clean.columns)}")
        print(f"📊 Final shape: {df_clean.shape}")

        return df_clean

    def scale_numerical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale numerical features using StandardScaler

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features

        Returns:
            Tuple: Scaled training features and optionally scaled test features
        """
        # Identify numerical features (excluding binary encoded features)
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        binary_features = [col for col in numerical_features if X_train[col].nunique() == 2]
        features_to_scale = [col for col in numerical_features if col not in binary_features]

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy() if X_test is not None else None

        if features_to_scale:
            # Fit scaler on training data
            X_train_scaled[features_to_scale] = self.scaler.fit_transform(X_train[features_to_scale])

            # Transform test data if provided
            if X_test is not None:
                X_test_scaled[features_to_scale] = self.scaler.transform(X_test[features_to_scale])

            print(f"✅ Scaled features: {features_to_scale}")

        return X_train_scaled, X_test_scaled

    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = 'Survived') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline

        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name

        Returns:
            Tuple: Preprocessed features and target
        """
        print("🚀 Starting preprocessing pipeline...")
        print("="*40)

        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df)

        # Step 2: Extract title from names
        df_processed = self.extract_title_from_name(df_processed)

        # Step 3: Create family features
        df_processed = self.create_family_features(df_processed)

        # Step 4: Create binned features
        df_processed = self.create_binned_features(df_processed)

        # Step 5: Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)

        # Step 6: Remove unnecessary features
        df_processed = self.remove_unnecessary_features(df_processed)

        # Step 7: Separate features and target
        if target_col in df_processed.columns:
            X = df_processed.drop(target_col, axis=1)
            y = df_processed[target_col]
        else:
            X = df_processed
            y = None

        # Store feature names
        self.feature_names = X.columns.tolist()

        print("🎉 Preprocessing pipeline completed!")
        print(f"📊 Final dataset shape: {X.shape}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict() if y is not None else 'N/A'}")

        return X, y

def preprocess_titanic_data(filepath: str, target_col: str = 'Survived') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to preprocess Titanic data

    Args:
        filepath (str): Path to the CSV file
        target_col (str): Target column name

    Returns:
        Tuple: Preprocessed features and target
    """
    preprocessor = TitanicDataPreprocessor()

    # Load data
    df = preprocessor.load_and_validate_data(filepath)

    # Run preprocessing pipeline
    X, y = preprocessor.preprocess_pipeline(df, target_col)

    return X, y, preprocessor

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Titanic Data Preprocessor...")

    try:
        X, y, preprocessor = preprocess_titanic_data('Titanic-Dataset.csv')
        print("\n✅ Preprocessing test successful!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature names: {X.columns.tolist()}")

    except Exception as e:
        print(f"❌ Preprocessing test failed: {str(e)}")
