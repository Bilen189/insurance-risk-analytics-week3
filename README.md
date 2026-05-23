Insurance Risk Analytics
 Project Overview

This project analyzes insurance risk data for AlphaCare Insurance Solutions (ACIS).

The goal is to:

Understand risk patterns in insurance customers
Identify factors influencing insurance charges (risk)
Perform statistical analysis and hypothesis testing
Build predictive models for insurance pricing

This helps ACIS improve risk-based pricing strategies.

 Dataset

The dataset contains the following features:

age
sex
bmi
children
smoker
region
charges (insurance cost / risk indicator)

We use charges as a proxy for insurance risk and claim cost.

 Key Metrics
Charges → represents insurance cost (risk level)
Higher charges = higher risk customers
 Project Tasks
Task 1: Exploratory Data Analysis (EDA)
Data cleaning and exploration
Risk analysis by demographic factors
Visualization of patterns in insurance charges
Task 2: Data Version Control (DVC)
Dataset versioning using DVC
Reproducible data pipeline
Local remote storage setup
Task 3: Hypothesis Testing (Upcoming)
Statistical comparison of groups (e.g. smokers vs non-smokers)
Identify significant risk factors
Task 4: Predictive Modeling (Upcoming)
Predict insurance charges using ML models
Linear Regression, Random Forest, XGBoost
 Setup Instructions
1. Clone repository
git clone <repo-url>
cd insurance-risk-analytics-week3
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run notebook
jupyter notebook

 Data Version Control (DVC)

This project uses DVC to track dataset versions.

Initialize DVC
dvc init
Track dataset
dvc add data/insurance.csv
Configure storage
dvc remote add -d localstorage C:\dvc-storage
Push data
dvc push
Check status
dvc status
 Tools Used
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
DVC
Git & GitHub
 Author

Insurance Risk Analytics Project — ACIS Training