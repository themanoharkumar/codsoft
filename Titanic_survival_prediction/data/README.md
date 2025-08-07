# Titanic Dataset Information

## Overview
This directory contains the Titanic dataset used for survival prediction analysis as part of the CODSOFT Data Science Internship (Task 1).

## Dataset Description

The Titanic dataset contains information about passengers aboard the RMS Titanic, including their survival status after the disaster on April 15, 1912.

### Files Structure
```
data/
├── raw/                    # Original, unmodified data
│   └── Titanic-Dataset.csv # Main dataset file
├── processed/              # Cleaned and preprocessed data
│   └── (generated during preprocessing)
└── README.md              # This file
```

## Dataset Schema

| Column | Data Type | Description | Missing Values |
|--------|-----------|-------------|----------------|
| PassengerId | int64 | Unique identifier for each passenger | 0 |
| Survived | int64 | Survival status (0 = No, 1 = Yes) | 0 |
| Pclass | int64 | Passenger class (1 = First, 2 = Second, 3 = Third) | 0 |
| Name | object | Full passenger name with title | 0 |
| Sex | object | Gender (male/female) | 0 |
| Age | float64 | Age in years | 177 (19.9%) |
| SibSp | int64 | Number of siblings/spouses aboard | 0 |
| Parch | int64 | Number of parents/children aboard | 0 |
| Ticket | object | Ticket number | 0 |
| Fare | float64 | Passenger fare paid | 0 |
| Cabin | object | Cabin number | 687 (77.1%) |
| Embarked | object | Port of embarkation (C/Q/S) | 2 (0.2%) |

## Key Statistics

- **Total Records**: 891 passengers
- **Target Variable**: Survived (0 = Did not survive, 1 = Survived)
- **Survival Rate**: 38.4% (342 survivors out of 891 passengers)
- **Missing Data**: Age (19.9%), Cabin (77.1%), Embarked (0.2%)

## Feature Descriptions

### Passenger Class (Pclass)
- **1**: First Class (upper class)
- **2**: Second Class (middle class)
- **3**: Third Class (lower class)

### Embarkation Ports (Embarked)
- **C**: Cherbourg, France
- **Q**: Queenstown, Ireland (now Cobh)
- **S**: Southampton, England

### Family Relations
- **SibSp**: Number of siblings and spouses aboard
  - Sibling: brother, sister, stepbrother, stepsister
  - Spouse: husband, wife (mistresses and fiancés ignored)
- **Parch**: Number of parents and children aboard
  - Parent: mother, father
  - Child: daughter, son, stepdaughter, stepson
  - Some children traveled only with a nanny, therefore parch=0

## Data Quality Issues

### Missing Values
1. **Age (177 missing)**: Can be imputed using median by passenger class and gender
2. **Cabin (687 missing)**: High missingness, can be converted to binary "HasCabin" feature
3. **Embarked (2 missing)**: Can be filled with mode (most common port)

### Data Preprocessing Recommendations

1. **Handle Missing Values**
   - Age: Impute with median grouped by Pclass and Sex
   - Cabin: Create binary feature indicating presence
   - Embarked: Fill with mode ('S')

2. **Feature Engineering**
   - Extract title from Name (Mr, Mrs, Miss, Master, etc.)
   - Create FamilySize = SibSp + Parch + 1
   - Create IsAlone binary feature
   - Bin Age and Fare into categories
   - Encode categorical variables

3. **Feature Selection**
   - Drop irrelevant features: PassengerId, Name, Ticket
   - Consider dropping Cabin due to high missingness
   - Encode Sex as binary (0/1)

## Survival Analysis Insights

Based on historical analysis, key factors affecting survival include:

1. **Gender**: Females had significantly higher survival rates (~74% vs ~19% for males)
2. **Passenger Class**: First class passengers had higher survival rates (~63% vs ~24% for third class)
3. **Age**: Children had higher survival rates
4. **Family Size**: Small families (2-4 members) had better survival rates
5. **Embarkation**: Passengers from Cherbourg had higher survival rates

## Usage Instructions

### Loading the Data
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/Titanic-Dataset.csv')

# Display basic information
print(df.info())
print(df.describe())
```

### Preprocessing Pipeline
Use the provided `data_preprocessing.py` module:

```python
from data_preprocessing import preprocess_titanic_data

# Preprocess the data
X, y, preprocessor = preprocess_titanic_data('data/raw/Titanic-Dataset.csv')
```

## Data Source

The Titanic dataset is a classic machine learning dataset, commonly used for binary classification problems. The original data comes from the British Board of Trade's official inquiry into the disaster.

## File Formats

- **CSV Format**: Comma-separated values with header row
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Missing Values**: Represented as NaN or empty strings

## License and Attribution

This dataset is publicly available and commonly used for educational purposes. It has been used in various machine learning competitions and tutorials.

## Contact

For questions about the dataset or preprocessing pipeline, please refer to the main project documentation or contact the project maintainer.

---

**Last Updated**: August 2025  
**Dataset Version**: 1.0  
**Project**: CODSOFT Data Science Internship - Task 1