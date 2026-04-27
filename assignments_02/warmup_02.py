"""
Python 200: Assignment 02 - scikit-learn Warmups
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# scikit-learn Question 1: Linear Regression

print("--- Scikit-Learn Q1 Output ---")

# 1. Prep Data
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. Create model
lin_reg = LinearRegression()

# 3. FIT model to training data
lin_reg.fit(years, salary)

# 4. PREDICT on new data
X_new = np.array([[4], [8]])
predictions = lin_reg.predict(X_new)

# 5. Output Results
print(f"Slope (coef_[0]): {lin_reg.coef_[0]:.2f}")
print(f"Intercept: {lin_reg.intercept_:.2f}")
print(f"Predicted salary for 4 years: ${predictions[0]:,.2f}")
print(f"Predicted salary for 8 years: ${predictions[1]:,.2f}\n")



# scikit-learn Question 2: Reshaping 1D to 2D

print("--- Scikit-Learn Q2 Output ---")

x = np.array([10, 20, 30, 40, 50])
print(f"Original 1D shape: {x.shape}")

# Reshape using -1 (which means "figure out the number of rows automatically")
# and 1 (which means "give me exactly 1 column").
x_2d = x.reshape(-1, 1)
print(f"New 2D shape: {x_2d.shape}\n")

"""
COMMENT: Why does scikit-learn need X to be 2D?
Scikit-learn always expects data in a standard "spreadsheet" format, where 
rows represent individual samples (data points) and columns represent features 
(variables). A 1D array is ambiguous—it could be 5 samples with 1 feature, 
or 1 sample with 5 features.
"""



# scikit-learn Question 3: K-Means Clustering

print("--- Scikit-Learn Q3 Output ---")

# 1. Prep Synthetic Data
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 2.
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. fit+ predict
labels = kmeans.fit_predict(X_clusters)

# Output results
print("Cluster Centers:")
print(kmeans.cluster_centers_)

counts = np.bincount(labels)
print(f"\nPoints per cluster: {counts}")

# 4. plot
plt.figure(figsize=(8, 6))

# all points, colored per cluster
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', alpha=0.7)

# center coordinates
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    c='black', 
    marker='X', 
    s=200, 
    label='Centers'
)

plt.title('K-Means Clustering (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()

# Save
plt.savefig('outputs/kmeans_clusters.png')
plt.clf() # Clear figure so it doesn't overlap with future plots!

print("Saved cluster plot to outputs/kmeans_clusters.png")


# Linear Regression Setup

from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# Linear Regression Q1: Data Exploration

print("\n--- Linear Regression Q1 Output ---")

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.8, edgecolors="k")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")
plt.tight_layout()
plt.savefig("outputs/cost_vs_age.png")
plt.clf()
print("Saved outputs/cost_vs_age.png")

"""
COMMENT: Are there two distinct groups visible? What does that suggest?
There are two distinct, parallel "bands" of data points. 
Red band represents the smokers, and the lower blue band represents 
the non-smokers. This suggests that the 'smoker' variable is a big predictor of medical costs, adding a large flat premium on top of the gradual 
increase caused by age.
"""


# Linear Regression Q2: Train/Test Split

print("\n--- Linear Regression Q2 Output ---")

# Age to 2D
X_age = age.reshape(-1, 1)
y = cost

X_train_age, X_test_age, y_train, y_test = train_test_split(
    X_age, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train_age.shape}")
print(f"X_test shape:  {X_test_age.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")



# Linear Regression Q3: Single Feature Model

print("\n--- Linear Regression Q3 Output ---")

model_age = LinearRegression()
model_age.fit(X_train_age, y_train)

# Predict
y_pred_age = model_age.predict(X_test_age)

# Calculate metrics
rmse_age = np.sqrt(np.mean((y_pred_age - y_test) ** 2))
r2_age = model_age.score(X_test_age, y_test)

print(f"Slope (Age): {model_age.coef_[0]:.2f}")
print(f"Intercept:   {model_age.intercept_:.2f}")
print(f"RMSE:        {rmse_age:.2f}")
print(f"R-squared:   {r2_age:.4f}")

"""
COMMENT: What does the slope mean in plain English?
It represents the estimated increase in annual medical costs for 
each additional year of life. Based on this single-feature model, every 
year older a patient gets, their expected medical cost increases by that 
slope amount.
"""


# Linear Regression Q4: Multiple Features

print("\n--- Linear Regression Q4 Output ---")

# Combine age and smoker
X_full = np.column_stack([age, smoker])

# Split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Fit
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

# Predict
r2_full = model_full.score(X_test_full, y_test_full)

print(f"R-squared (Age + Smoker): {r2_full:.4f}")
print(f"Age coefficient:    {model_full.coef_[0]:.2f}")
print(f"Smoker coefficient: {model_full.coef_[1]:.2f}")

"""
COMMENT: Does adding the smoker flag help? What does its coefficient mean?
Smoker flag causes the R-squared value to jump significantly 
(closer to 1.0), meaning the model explains much more of the variance in the data.
The smoker coefficient represents the added cost of being a smoker. 
It means a smoker's annual medical cost is estimated to be 
that exact coefficient amount higher than a non-smoker of the exact same age.
"""


# Linear Regression Q5: Predicted vs Actual Plot

print("\n--- Linear Regression Q5 Output ---")

# Generate
y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7, edgecolors="k", color="mediumseagreen")

# Create
min_val = min(min(y_pred_full), min(y_test_full))
max_val = max(max(y_pred_full), max(y_test_full))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")

plt.title("Predicted vs Actual Medical Costs")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.clf()
print("Saved outputs/predicted_vs_actual.png")

"""
COMMENT: What does it mean when a point falls above or below the diagonal?
A point ABOVE the diagonal means the actual cost on the y-axis was higher 
than what the model predicted on the x-axis (the model under-predicted). 
A point BELOW the diagonal means the actual cost was lower than what the 
model predicted (the model over-predicted).
"""