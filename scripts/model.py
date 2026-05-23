# -----------------------------------------------
# model.py
# -----------------------------------------------
# Purpose: Handles everything related to model training and evaluation.
# Four responsibilities:
#   1. Split data into training and testing sets
#   2. Train four regression models on the training set
#   3. Evaluate each model on the test set using MAE, MSE, R²
#   4. Plot bar charts comparing model performance
# -----------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# -----------------------------------------------
# FUNCTION 1: Split Data into Train and Test Sets
# -----------------------------------------------
# Input:
#   X            — feature matrix (all input columns)
#   y            — target column (what we are predicting: charges)
#   test_size    — fraction of data kept for testing (default 20%)
#   random_state — seed for reproducibility (same split every run)
# Output: X_train, X_test, y_train, y_test
#
# WHY SPLIT THE DATA?
#   If you train and test on the same data, your model will appear accurate
#   because it has already "seen" the answers. This is called overfitting.
#   The test set simulates NEW, unseen customers — giving you an honest
#   measure of how the model performs in the real world.
#
#   80% training / 20% testing is the standard split.
#   random_state=42 ensures everyone gets the same split (reproducibility).
# -----------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------
# FUNCTION 2: Train All Four Models
# -----------------------------------------------
# Input:  X_train, y_train — the training portion of the data
# Output: four trained model objects (lr, dt, rfr, xgb)
#
# HOW TRAINING WORKS:
#   .fit(X_train, y_train) is the learning step.
#   Each model looks at the features (X_train) and their known outcomes (y_train)
#   and finds the mathematical relationship that best maps inputs → outputs.
#   After .fit(), the model stores learned parameters internally and is
#   ready to make predictions on new data using .predict().
# -----------------------------------------------
def train_models(X_train, y_train):

    # -----------------------------------------------
    # MODEL 1: Linear Regression (Parametric / Statistical)
    # -----------------------------------------------
    # Fits a straight line (or hyperplane in multiple dimensions):
    #   charges = β₀ + β₁·age + β₂·bmi + β₃·smoker + ...
    # The model learns β coefficients that minimise the sum of squared errors.
    #
    # STRENGTHS:
    #   - Fastest to train, easiest to interpret
    #   - Each coefficient has a clear meaning: "1 year of age adds £X to charges"
    #   - Works well when the relationship between inputs and output is roughly linear
    #
    # WEAKNESSES:
    #   - Cannot capture non-linear relationships (e.g., charges spike non-linearly for smokers)
    #   - Assumes errors are normally distributed and variance is constant
    #   - Sensitive to outliers
    #
    # USE AS: the baseline model — if a complex model cannot beat this, something is wrong
    # -----------------------------------------------
    lr_model = LinearRegression()

    # -----------------------------------------------
    # MODEL 2: Decision Tree Regressor (Nonparametric)
    # -----------------------------------------------
    # Splits the data into branches using if/else rules:
    #   "If smoker=1 AND age > 42 → predict £28,000"
    #   "If smoker=0 AND bmi < 30 → predict £6,000"
    # The tree grows by finding the split at each step that most reduces prediction error.
    #
    # STRENGTHS:
    #   - Captures non-linear relationships and interactions between features
    #   - Highly interpretable — you can draw and read the tree
    #   - Handles mixed data types (numbers and categories) well
    #   - No scaling needed (splits are based on thresholds, not distances)
    #
    # WEAKNESSES:
    #   - Prone to overfitting — a deep tree memorises training data perfectly
    #     but performs poorly on new data (high variance)
    #   - Sensitive to small changes in data — different random seed can give a
    #     completely different tree structure
    #
    # random_state=42 ensures reproducible splits in the tree-building process
    # -----------------------------------------------
    dt_model = DecisionTreeRegressor(random_state=42)

    # -----------------------------------------------
    # MODEL 3: Random Forest Regressor (Ensemble / Nonparametric)
    # -----------------------------------------------
    # Builds many decision trees independently (default: 100 trees).
    # Each tree is trained on a random subset of rows AND a random subset of features.
    # Final prediction = AVERAGE of all trees' predictions.
    #
    # WHY THIS IS BETTER THAN A SINGLE TREE:
    #   A single tree overfits — it memorises noise. When you average 100 trees,
    #   each trained on slightly different data, the noise cancels out.
    #   This is called the "wisdom of crowds" — many imperfect predictors
    #   combined become a very accurate predictor.
    #
    # STRENGTHS:
    #   - Much more accurate than a single decision tree
    #   - Robust to overfitting (ensemble averaging reduces variance)
    #   - Handles missing data and outliers better than linear models
    #   - Provides feature importance scores
    #
    # WEAKNESSES:
    #   - Slower to train (100+ trees vs 1)
    #   - Less interpretable than a single tree
    #   - Harder to explain a specific prediction (though SHAP helps)
    # -----------------------------------------------
    rfr_model = RandomForestRegressor(random_state=42)

    # -----------------------------------------------
    # MODEL 4: XGBoost Regressor (Gradient Boosting / Nonparametric)
    # -----------------------------------------------
    # Builds trees SEQUENTIALLY — each new tree learns from the ERRORS
    # (residuals) of all previous trees combined.
    #
    # HOW IT DIFFERS FROM RANDOM FOREST:
    #   Random Forest: trees are built independently and averaged (parallel)
    #   XGBoost:       each tree corrects the mistakes of the last (sequential)
    #
    # Example of boosting in 3 steps:
    #   Tree 1: Predicts £10,000, actual is £15,000 → error = £5,000
    #   Tree 2: Focuses on that £5,000 error, corrects to £4,000 → cumulative = £14,000
    #   Tree 3: Corrects remaining £1,000 → cumulative = £15,000
    #
    # STRENGTHS:
    #   - State-of-the-art accuracy on tabular (structured) data
    #   - Won more Kaggle competitions than any other algorithm
    #   - Built-in regularisation (L1 and L2) to prevent overfitting
    #   - Handles missing values internally
    #
    # WEAKNESSES:
    #   - Many hyperparameters to tune (learning rate, max depth, n_estimators, etc.)
    #   - Can overfit if not tuned carefully
    #   - Slower than linear regression (but faster than deep neural networks)
    # -----------------------------------------------
    xgb_model = xgb.XGBRegressor(random_state=42)

    # -----------------------------------------------
    # TRAINING (the .fit() step for all four models)
    # -----------------------------------------------
    # Each .fit() call is the actual "learning" step.
    # The model reads X_train (features) and y_train (targets)
    # and updates its internal parameters to minimise prediction error.
    # After this step, each model is ready to predict on unseen data.
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rfr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return lr_model, dt_model, rfr_model, xgb_model


# -----------------------------------------------
# FUNCTION 3: Evaluate a Single Model
# -----------------------------------------------
# Input:
#   model   — a trained model object
#   X_test  — feature matrix from the test set (NOT seen during training)
#   y_test  — true target values from the test set
# Output: mae, mse, r2 scores and the predicted values (y_pred)
#
# HOW EVALUATION WORKS:
#   model.predict(X_test) → model generates predictions for each test row
#   We compare those predictions against the actual known values (y_test)
#   to calculate how accurate the model is on data it has NEVER seen before.
# -----------------------------------------------
def evaluate_model(model, X_test, y_test):

    # Generate predictions on the held-out test set
    y_pred = model.predict(X_test)

    # -----------------------------------------------
    # METRIC 1: MAE — Mean Absolute Error
    # -----------------------------------------------
    # Formula: average of |y_actual - y_predicted| for every row
    # Units: same as the target variable (e.g., pounds, dollars)
    # Interpretation: "On average, our predictions are off by £X"
    #
    # PROS: intuitive, same unit as the target, not distorted by outliers
    # CONS: does not punish large errors — a £1,000 error counts the same
    #       per-unit as a £100 error (just 10x, not 100x)
    # -----------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)

    # -----------------------------------------------
    # METRIC 2: MSE — Mean Squared Error
    # -----------------------------------------------
    # Formula: average of (y_actual - y_predicted)² for every row
    # Units: squared units of the target (e.g., pounds²) — hard to interpret directly
    # Interpretation: penalises large errors much more heavily than small ones
    #
    # WHY USE IN INSURANCE:
    #   A £10,000 prediction error is not just 10x worse than a £1,000 error —
    #   it is 100x worse in MSE terms. This reflects the real business impact:
    #   severely underpricing a high-risk policy has catastrophic consequences.
    #
    # RMSE (Root MSE) = √MSE brings it back to the original unit scale.
    # -----------------------------------------------
    mse = mean_squared_error(y_test, y_pred)

    # -----------------------------------------------
    # METRIC 3: R² — Coefficient of Determination
    # -----------------------------------------------
    # Formula: 1 - (sum of squared residuals / total sum of squares)
    # Range: 0 to 1 (can be negative for very bad models)
    # Interpretation: "The model explains X% of the variation in charges"
    #
    # Examples:
    #   R² = 0.88 → the model explains 88% of variance in charges (good)
    #   R² = 0.50 → only 50% explained — 50% is still unexplained noise
    #   R² = 0.00 → model does no better than just predicting the mean
    #   R² < 0   → model is worse than predicting the mean (something is very wrong)
    #
    # R² is the most common single-number summary of regression model performance.
    # -----------------------------------------------
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred


# -----------------------------------------------
# FUNCTION 4: Plot Model Comparison Bar Charts
# -----------------------------------------------
# Input:
#   models     — list of model name strings (for x-axis labels)
#   mae_scores — list of MAE values, one per model
#   mse_scores — list of MSE values, one per model
#   r2_scores  — list of R² values, one per model
# Output: three bar charts displayed side by side
#
# WHY VISUALISE:
#   A table of numbers is harder to read at a glance than a bar chart.
#   Visualising metrics makes it immediately obvious which model wins
#   on each metric — especially useful for presentations.
# -----------------------------------------------
def plot_metrics(models, mae_scores, mse_scores, r2_scores):

    # -----------------------------------------------
    # CHART 1: MAE Comparison
    # Lower bar = better (fewer prediction errors on average)
    # -----------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(models, mae_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of MAE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------
    # CHART 2: MSE Comparison
    # Lower bar = better
    # MSE values are typically much larger than MAE (squared units)
    # -----------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE Scores — Lower is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------
    # CHART 3: R² Comparison
    # Higher bar = better (explains more variance)
    # Maximum possible value = 1.0 (perfect model)
    # -----------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(models, r2_scores, color='salmon')
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.title('Comparison of R² Scores — Higher is Better')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
