# Titanic Survival Prediction - CODSOFT Internship
## 📝 Project Overview

This project is part of the **CODSOFT Data Science Internship** (Task 1). The goal is to predict the survival of passengers on the Titanic disaster using machine learning techniques. This is a classic binary classification problem that analyzes passenger data such as age, gender, ticket class, and fare to determine survival probability.

## 🎯 Objective

Build a robust machine learning model that can predict whether a passenger on the Titanic survived or not based on their demographic and ticket information with high accuracy.

## 📊 Dataset Information

The dataset contains information about individual Titanic passengers including:

- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes) - **Target Variable**
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard  
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

**Dataset Shape**: 891 passengers, 12 features

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CODSOFT-Titanic-Survival-Prediction.git
   cd CODSOFT-Titanic-Survival-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv titanic_env
   source titanic_env/bin/activate  # On Windows: titanic_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete project**
   ```bash
   python titanic_survival_prediction.py
   ```

### Alternative: Run modular scripts
```bash
# Train models
python scripts/train.py

# Make predictions
python scripts/predict.py

# Evaluate models
python scripts/evaluate.py
```

## 📁 Project Structure

```
CODSOFT-Titanic-Survival-Prediction/
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
├── 📄 config.py                         # Configuration settings
├── 🐍 titanic_survival_prediction.py    # Complete standalone script
│
├── 📁 data/                             # Data directory
│   ├── 📁 raw/                          # Original dataset
│   ├── 📁 processed/                    # Cleaned/processed data
│   └── 📄 README.md                     # Data description
│
├── 📁 models/                           # Saved model files
├── 📁 results/                          # Output results
│   ├── 📁 figures/                      # Generated plots
│   └── 📁 reports/                      # Analysis reports

## 🔧 Features & Methodology

### Data Preprocessing
- Handle missing values (Age, Cabin, Embarked)
- Remove irrelevant features (PassengerId, Name, Ticket)
- Encode categorical variables
- Feature scaling and normalization

### Feature Engineering
- **Title Extraction**: Extract titles from passenger names (Mr, Mrs, Miss, etc.)
- **Family Size**: Combine SibSp and Parch to create family size feature
- **Is Alone**: Binary feature indicating if passenger traveled alone
- **Age Binning**: Categorize ages into groups
- **Fare Binning**: Categorize fares into quartiles

### Machine Learning Models
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method for better accuracy
- **Support Vector Machine (SVM)**: Non-linear classification
- **Gradient Boosting**: Advanced ensemble method
- **Neural Network**: Deep learning approach

### Model Evaluation
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, Recall, F1-score
- **ROC Curve**: Receiver Operating Characteristic
- **Feature Importance**: Analysis of most predictive features

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 82.1% | 0.79 | 0.74 | 0.76 |
| Random Forest | 83.8% | 0.81 | 0.77 | 0.79 |
| SVM | 82.7% | 0.80 | 0.75 | 0.77 |
| Gradient Boosting | 84.2% | 0.82 | 0.78 | 0.80 |

**Best Model**: Gradient Boosting with 84.2% accuracy

### Key Insights
- **Gender** is the strongest predictor (women had higher survival rates)
- **Passenger Class** significantly impacts survival (1st class > 2nd class > 3rd class)
- **Age** and **Family Size** are also important factors
- Passengers who embarked from **Cherbourg** had higher survival rates

## 📊 Visualizations

The project generates various visualizations including:
- Survival rates by gender and passenger class
- Age distribution of survivors vs non-survivors
- Correlation heatmap of features
- Feature importance plots
- Model performance comparisons
- Confusion matrices

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development
- **Git**: Version control

## 📋 Requirements

See `requirements.txt` for complete list of dependencies:

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```
## 👨‍💻 Author

**[Manohar]**
- GitHub: [@themanoharkumar](https://github.com/themanoharkumar)
- LinkedIn: [Manohar singh](https://www.linkedin.com/in/manohar-singh-b28189324/)
- Email: themanoharkumar983@gmail.com

## 🙏 Acknowledgments

- **CODSOFT** for the internship opportunity
- Kaggle for providing the Titanic dataset
- The data science community for inspiration and resources


**#CODSOFT #DataScience #MachineLearning #Titanic #Python #Internship**