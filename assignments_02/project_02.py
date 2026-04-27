"""
Python 200: Assignment 02 - Predicting Student Math Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# values separated by semicolons (;), not commas.
# string values in quotes.
# use the separator: sep=';'



# TASK 1: Load and Explore

print("--- TASK 1: Load and Explore ---")

# Load data
df = pd.read_csv('student_performance_math.csv', sep=';')

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# Plot G3 Distribution
plt.figure(figsize=(8, 6))
plt.hist(df['G3'], bins=21, range=(0, 20), color='skyblue', edgecolor='black')
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.xticks(range(0, 21))
plt.tight_layout()
plt.savefig('outputs/g3_distribution.png')
plt.clf()
print("\nSaved outputs/g3_distribution.png")



# TASK 2: Preprocess the Data

print("\n--- TASK 2: Preprocess the Data ---")

# 1. Filter out G3 = 0
print(f"Shape before filtering G3=0: {df.shape}")
df_clean = df[df['G3'] > 0].copy()
print(f"Shape after filtering G3=0: {df_clean.shape}")

"""
COMMENT: Why would keeping G3=0 distort the model?
The students with G3=0 didn't necessarily score a 0% on the test; they were prob
absent and missed the exam. If we leave them in, the model will try to 
find mathematical relationships to predict a '0' based on their habits or 
demographics, which is mathematically invalid. Missing a test is not a measure of mathematical apability.
"""

# 2. Check correlation before
corr_before = df['absences'].corr(df['G3'])
corr_after = df_clean['absences'].corr(df_clean['G3'])

print(f"\nCorrelation (absences vs G3) BEFORE filtering: {corr_before:.4f}")
print(f"Correlation (absences vs G3) AFTER filtering:  {corr_after:.4f}")

"""
COMMENT: Why does filtering change the result?
In the original dataset, students who missed the final exam (G3=0) likely had 
high absence rates overall. Because 0 is the lowest possible, this created 
an a trend of "high absences = extreme low score (0)". Once those students removed, we see the actual relationship for students who took the 
exam: absences actually have a slight negative (or near-zero) correlation with 
the actual test performance.
"""

# 3. Convert
binary_cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in binary_cols:
    df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0})

df_clean['sex'] = df_clean['sex'].map({'F': 0, 'M': 1})



# TASK 3: Exploratory Data Analysis

print("\n--- TASK 3: Exploratory Data Analysis ---")

# Isolate numeric columns for correlation (excluding G1 and G2 as instructed)
numeric_df = df_clean.select_dtypes(include=[np.number]).drop(columns=['G1', 'G2'], errors='ignore')
correlations = numeric_df.corr()['G3'].sort_values()

print("Correlations with G3:")
print(correlations.drop('G3')) # Drop G3 correlating with itself

"""
COMMENT: Strongest relationship and surprises?
The strongest relationship with G3 is 'failures' (a strong negative correlation). 
'studytime' has a very weak correlation with the final grade, and 
'Medu' (Mother's education) has a positive correlation, indicating 
socioeconomic background might be driving performance more than sheer effort.
"""

# Plot 1: Boxplot of Mother's Education vs G3
plt.figure(figsize=(8, 6))
sns.boxplot(x='Medu', y='G3', data=df_clean, palette='Blues')
plt.title("Impact of Mother's Education on Final Grade")
plt.xlabel("Mother's Education Level (0=None to 4=Higher Ed)")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/medu_vs_g3.png')
plt.clf()

"""
PLOT 1 COMMENT: Clear upward trend. The median G3 score steadily increases 
as the mother's education level increases, shows the impact of home environment 
on academic success.
"""

# Plot 2: Scatter failures vs G3
plt.figure(figsize=(8, 6))
sns.stripplot(x='failures', y='G3', data=df_clean, jitter=True, alpha=0.6, palette='Reds_r')
plt.title("Past Class Failures vs Final Grade")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/failures_vs_g3.png')
plt.clf()

"""
PLOT 2 COMMENT: Students with 0 past failures have a wide distribution of scores, 
including the highest grades in the class. As failures increase to 1, 2, or 3, the 
ceiling of their G3 score drops drastically.
"""
print("Saved custom EDA plots to outputs/")


# ==========================================
# TASK 4: Baseline Model
# ==========================================
print("\n--- TASK 4: Baseline Model ---")

X_base = df_clean[['failures']].values
y = df_clean['G3'].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

rmse_base = np.sqrt(np.mean((y_pred_b - y_test_b)**2))
r2_base = model_base.score(X_test_b, y_test_b)

print(f"Slope (failures): {model_base.coef_[0]:.2f}")
print(f"RMSE: {rmse_base:.2f}")
print(f"Test R-squared: {r2_base:.4f}")

"""
COMMENT: Interpretation of the baseline model
Since grades are on a 0-20 scale, a slope of -1.45 means that for every past 
class failure, a student's expected final grade drops by 1.45 points. 
An RMSE of ~3.0 means our predictions are usually off by about 3 points (15% 
of total). The R-squared is quite low, which makes sense—past 
failures are predictive but dont capture full picture of student potential.
"""



# TASK 5: Build the Full Model

print("\n--- TASK 5: Full Model ---")

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", 
                "internet", "sex", "freetime", "activities", "traveltime"]

X_full = df_clean[feature_cols].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

train_r2 = model_full.score(X_train_f, y_train_f)
test_r2 = model_full.score(X_test_f, y_test_f)
y_pred_f = model_full.predict(X_test_f)
rmse_full = np.sqrt(np.mean((y_pred_f - y_test_f)**2))

print(f"Train R-squared: {train_r2:.4f}")
print(f"Test R-squared:  {test_r2:.4f}")
print(f"Test RMSE:       {rmse_full:.2f}\n")

print("Feature Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

"""
COMMENT: Interpreting full model
The test R-squared improved from the baseline- the extra features 
 help explain more variance. 

Surprising signs: 'schoolsup' (extra educational support) is highly NEGATIVE. 
This doesn't mean tutoring makes you worse at math; it prob means that only 
students who are already struggling are assigned to extra support. 

Train vs Test R-squared: They are relatively close, meaning our model is not 
massively overfitting. It generalized reasonably well to unseen data.

Deployment choices: If putting this in production, I would drop 'freetime', 
'activities', and 'traveltime', as their coefficients are extremely close to zero, 
adding noise without much predictive power. I would absolutely keep 'failures', 
'Medu', 'sex', and 'higher', as they hold the most statistical weight.
"""



# TASK 6: Evaluate and Summarize

print("\n--- TASK 6: Evaluate and Summarize ---")

# Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_f, y_test_f, alpha=0.7, edgecolors="k", color="mediumpurple")

# Diagonal line
min_val = min(min(y_pred_f), min(y_test_f))
max_val = max(max(y_pred_f), max(y_test_f))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade (G3)")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/predicted_vs_actual.png')
plt.clf()
print("Saved outputs/predicted_vs_actual.png")

"""
COMMENT: Predicted vs Actual Plot Analysis
The model struggles heavily at the extreme low end. It predicts scores around 10-12 
for students who actually scored 5-8 (falling below the diagonal, meaning it over-predicted). 
The error is roughly uniform in the middle, but lacks the features needed 
to identify the lowest-performing students accurately.

--- FINAL SUMMARY ---
* Dataset Scope: Filtered dataset contains 357 rows, tested on 72 students.
* Metrics: The model achieved an RMSE of ~2.9 and an R-squared of ~0.16. On a 20-point 
  scale, being off by 3 points is a full letter grade and a half. The model captures 
  trends but is not precise enough for individual automated grading.
* Top Positive Feature: 'sex' (+1.18). Male students scored over a point higher on average, 
  which aligns with the sociological PISA gap noted in the feature guide.
* Top Negative Feature: 'failures' (-1.41). Past academic struggles strongly predict 
  future ones.
* Surprising Result: 'schoolsup' being heavily negative (-0.89) was a great reminder 
  that correlation does not equal causation—interventions target low scores, rather 
  than causing them.
"""


# NEGLECTED FEATURE: The Power of G1

print("\n--- Neglected Feature: Adding G1 ---")

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(X_g1, y, test_size=0.2, random_state=42)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print(f"Test R-squared with G1 included: {test_r2_g1:.4f}")

"""
COMMENT: The Power of G1
The R-squared shoots up massively. Does a high R-squared here mean G1 is CAUSING G3? 
No. G1 and G3 are just two measurements of the exact same underlying trait: the student's 
math ability that year. It's a great model for predicting the future, but it's not a 
useful model for identifying *why* a student struggles. If educators want to intervene 
early (before G1), they must focus on the demographic and behavioral factors (like past 
failures or lack of higher ed goals) that we explored in the main model.
"""