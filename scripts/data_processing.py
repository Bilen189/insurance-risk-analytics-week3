# -----------------------------------------------
# data_processing.py
# -----------------------------------------------
# Purpose: Handles all data preparation steps before model training.
# Three responsibilities:
#   1. Load raw CSV data and remove duplicates
#   2. Encode categorical columns into numbers (two methods)
#   3. Scale numerical columns into a consistent range (three methods)
# -----------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np


# ------------------------------------------------
# FUNCTION 1: Load and Clean Data
# ------------------------------------------------
# Input:  filepath — path to the CSV file (string)
# Output: cleaned DataFrame with duplicate rows removed
# ------------------------------------------------
def load_and_clean_data(filepath):

    # Read the CSV file into a pandas DataFrame
    # A DataFrame is a table — rows are customers, columns are features
    data = pd.read_csv(filepath)

    # Remove exact duplicate rows
    # keep="first" means: if two rows are identical, keep the first one and delete the rest
    # WHY: duplicate records inflate the training set and bias the model toward
    # patterns that appear in duplicated rows rather than real patterns in the data
    data = data.drop_duplicates(keep="first")

    return data


# ------------------------------------------------
# FUNCTION 2: Encode Categorical Variables
# ------------------------------------------------
# Input:
#   method         — which encoding to use: 'labelEncoder' or 'oneHotEncoder'
#   dataframe      — the DataFrame to encode
#   columns_label  — list of columns to label-encode
#   columns_onehot — list of columns to one-hot encode
# Output: new DataFrame with categorical columns replaced by numbers
#
# WHY ENCODING IS NEEDED:
#   Machine learning models perform mathematical operations (multiplication,
#   addition, distance calculations). They cannot do math on text like "male"
#   or "southwest". Encoding converts text → numbers so the math works.
# ------------------------------------------------
def encoder(method, dataframe, columns_label, columns_onehot):

    # ---------------------------
    # METHOD 1: LABEL ENCODER
    # ---------------------------
    # Converts each unique category to a unique integer
    # Example: sex ["female", "male"] → [0, 1]
    # Example: region ["northeast","northwest","southeast","southwest"] → [0,1,2,3]
    #
    # WHEN TO USE:
    #   - When the categorical variable has an inherent order (e.g., "low", "medium", "high")
    #   - Or when using tree-based models (Decision Tree, Random Forest, XGBoost)
    #     which can handle arbitrary integer encodings without implying order
    #
    # CAUTION:
    #   - Do NOT use label encoding with linear regression on unordered categories.
    #     The model will assume 0 < 1 < 2 (ordered) which is wrong for e.g. regions.
    # ---------------------------
    if method == 'labelEncoder':

        df_lbl = dataframe.copy()
        # .copy() creates an independent duplicate — changes here do NOT affect
        # the original dataframe. Always copy before transforming to preserve raw data.

        for col in columns_label:

            label = LabelEncoder()

            # .fit() scans the column and records all unique categories
            # e.g., for 'sex': learns {"female": 0, "male": 1}
            label.fit(list(dataframe[col].values))

            # .transform() applies the learned mapping to every value in the column
            # "female" → 0, "male" → 1 for every row
            df_lbl[col] = label.transform(df_lbl[col].values)

        return df_lbl

    # ---------------------------
    # METHOD 2: ONE-HOT ENCODER
    # ---------------------------
    # Creates one new binary (0/1) column per category
    # Example: region ["northeast","northwest","southeast","southwest"]
    # Becomes: ohe_region_northwest (0/1), ohe_region_southeast (0/1), ohe_region_southwest (0/1)
    # (northeast is dropped — see drop_first below)
    #
    # WHEN TO USE:
    #   - When the categorical variable has NO inherent order (nominal data)
    #   - When using linear models, logistic regression, neural networks
    #     which would misinterpret integer labels as meaningful numbers
    #
    # drop_first=True:
    #   Drops the first category column to avoid the "dummy variable trap"
    #   The trap: if you have 4 regions and keep all 4 binary columns,
    #   knowing the first 3 tells you exactly what the 4th is → perfect multicollinearity
    #   This would make linear regression mathematically unsolvable.
    # ---------------------------
    elif method == 'oneHotEncoder':

        df_oh = dataframe.copy()

        df_oh = pd.get_dummies(
            data=df_oh,
            prefix='ohe',           # New columns are named: ohe_sex, ohe_smoker, etc.
            prefix_sep='_',         # Separator: ohe_sex_male (not ohe sex male)
            columns=columns_onehot, # Only encode these specific columns
            drop_first=True,        # Drop first category to avoid multicollinearity
            dtype='int8'            # Store as 8-bit integer (0 or 1) — saves memory
        )

        return df_oh


# ------------------------------------------------
# FUNCTION 3: Scale Numerical Variables
# ------------------------------------------------
# Input:
#   method          — which scaling method: 'standardScaler', 'minMaxScaler', 'npLog'
#   data            — the DataFrame to scale
#   columns_scaler  — list of numerical columns to scale
# Output: new DataFrame with scaled numerical columns
#
# WHY SCALING IS NEEDED:
#   Many models are sensitive to the magnitude of feature values.
#   Example: age ranges from 18–64, BMI from 15–53, charges from £1,000–£63,000.
#   Without scaling, 'charges' dominates the model simply because its numbers
#   are 1,000x larger — even if age is equally important.
#   Scaling puts all features on a level playing field.
# ------------------------------------------------
def scaler(method, data, columns_scaler):

    # ---------------------------
    # METHOD 1: STANDARD SCALER
    # ---------------------------
    # Transforms each value to: (value - mean) / standard_deviation
    # Result: the column has mean = 0 and standard deviation = 1
    #
    # Example: charges with mean=£13,270 and std=£12,110
    #   A charge of £25,380 → (25380 - 13270) / 12110 = +1.0
    #   A charge of £1,160  → (1160 - 13270)  / 12110 = -1.0
    #
    # WHEN TO USE:
    #   - Linear Regression, Logistic Regression, SVM, Neural Networks
    #   - Any model that uses gradient descent or distance calculations
    #   - When your data has outliers but you still want to preserve their direction
    # ---------------------------
    if method == 'standardScaler':

        Standard = StandardScaler()
        df_standard = data.copy()

        # fit_transform does two things in one call:
        # .fit()      → calculates mean and std from the data
        # .transform() → applies (x - mean) / std to every value
        df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])

        return df_standard

    # ---------------------------
    # METHOD 2: MIN-MAX SCALER
    # ---------------------------
    # Transforms each value to: (value - min) / (max - min)
    # Result: all values fall between 0 and 1
    #
    # Example: charges ranging £1,122 to £63,770
    #   A charge of £32,000 → (32000 - 1122) / (63770 - 1122) = 0.493
    #   A charge of £1,122  → (1122 - 1122)  / (63770 - 1122) = 0.0
    #   A charge of £63,770 → (63770 - 1122) / (63770 - 1122) = 1.0
    #
    # WHEN TO USE:
    #   - Neural Networks (inputs must be between 0 and 1 for stable training)
    #   - K-Nearest Neighbours, K-Means (distance-based — scale matters directly)
    #   - When you want to preserve the exact proportional relationship between values
    #
    # CAUTION: sensitive to outliers — one extreme value compresses everything else
    # ---------------------------
    elif method == 'minMaxScaler':

        MinMax = MinMaxScaler()
        df_minmax = data.copy()

        # fit_transform: learns min and max from data, then applies the formula
        df_minmax[columns_scaler] = MinMax.fit_transform(df_minmax[columns_scaler])

        return df_minmax

    # ---------------------------
    # METHOD 3: LOG TRANSFORMATION
    # ---------------------------
    # Applies natural logarithm: log(x) to each value
    # Result: compresses large values, expands small values, removes right skew
    #
    # Example: charges [1000, 5000, 10000, 50000]
    #   log([1000, 5000, 10000, 50000]) = [6.9, 8.5, 9.2, 10.8]
    #   The huge jump from 10,000 to 50,000 is compressed to just 1.6 units
    #
    # WHEN TO USE:
    #   - When your target variable is right-skewed (long tail on the right)
    #   - 'charges' and 'TotalClaims' in insurance are always right-skewed
    #   - Linear regression assumes normally distributed errors — log transformation
    #     helps satisfy this assumption
    #
    # IMPORTANT: log(0) is undefined and log(negative) is undefined.
    #   If your column has zeros, use np.log(x + 1) instead.
    # ---------------------------
    elif method == 'npLog':

        df_nplog = data.copy()

        # Apply natural log to every value in the specified columns
        df_nplog[columns_scaler] = np.log(df_nplog[columns_scaler])

        return df_nplog

    # Fallback: if an unrecognised method is passed, return the data unchanged
    return data
