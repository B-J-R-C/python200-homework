"""
Python 200: Assignment 03 - Warmup Exercises
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --- Setup ---
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ==========================================
# Preprocessing
# ==========================================

# --- Q1: Train/Test Split ---
print("--- Q1: Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}\n")


# --- Q2: StandardScaler ---
print("--- Q2: StandardScaler ---")
scaler = StandardScaler()

# Fit training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print mean of column
print("Means of X_train_scaled columns:")
print(np.mean(X_train_scaled, axis=0))

"""
COMMENT: Why do we fit the scaler on X_train only?
We fit the scaler on X_train to prevent "data leakage"—if we fit 
it on the entire dataset, information about the test set would "leak" into the training process, giving the model an unfair 
advantage and making evaluation metrics look better than they are.
"""


# K-Nearest Neighbors (KNN)
# Q1: KNN on Unscaled Data
print("\n--- Q1: KNN on Unscaled Data ---")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_unscaled):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_unscaled))


# Q2: KNN on Scaled Data
print("--- Q2: KNN on Scaled Data ---")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy (Scaled): {accuracy_score(y_test, y_pred_scaled):.4f}")

"""
COMMENT: Does scaling improve performance, hurt it, or make no difference?
For the Iris dataset, scaling makes little difference (and sometimes 
slightly hurts). This is because all four features are physical measurements (petal/sepal length/width) recorded in the exact 
same unit (centimeters) and have relatively similar ranges. KNN relies on 
distance- so scaling is important when units differ a lot. 
but its not crucial here.
"""


# Q3: Cross-Validation
print("\n--- Q3: Cross-Validation ---")
# Re-instantiating fresh model to be clear
knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

print(f"Fold Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

"""
COMMENT: Is this result more or less trustworthy than a single train/test split?
Cross-validation is MUCH more trustworthy. A single train/test split might get "lucky" 
or "unlucky" depending on exactly which rows ended up in the test set. By rotating 
through 5 different folds, we get a robust estimate of how the model will actually 
perform on completely unseen data.
"""


#Q4: Hyperparameter Tuning (k values)
print("\n--- Q4: Hyperparameter Tuning ---")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn_tune = KNeighborsClassifier(n_neighbors=k)
    # Using the unscaled training data as requested
    scores = cross_val_score(knn_tune, X_train, y_train, cv=5)
    print(f"k={k:2d} | Mean CV Accuracy: {scores.mean():.4f}")

"""
COMMENT: Which k would you choose and why?
I would choose the k that maximizes mean CV accuracy. If there is a draw , it is generally 
best to choose the higher k (up to a point). A higher k considers more 
neighbors, which smooths out the decision boundary and makes the model less 
likely to overfitting to noisy data points.
"""


# Classifier Evaluation

#Q1: Confusion Matrix
print("\n--- Classifier Evaluation Q1: Confusion Matrix ---")

# Calculate confusion matrix using predictions
cm = confusion_matrix(y_test, y_pred_unscaled)

# Display plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')

plt.title('KNN Confusion Matrix (k=5)')
plt.tight_layout()
plt.savefig('outputs/knn_confusion_matrix.png')
plt.clf() # Clear figure
print("Saved outputs/knn_confusion_matrix.png")

"""
COMMENT: Which pair of species does the model most often confuse?
Looking at the matrix, model usually confuses 'versicolor' and 'virginica'. 
'setosa' is always predicted perfectly because its physical measurements are 
completely distinct, but versicolor and virginica overlap a bit in their 
petal and sepal sizes, making them harder for the algorithm to separate.
"""


# The sklearn API: Decision Trees

# Q1: Decision Tree Classifier
print("\n--- Decision Tree Q1 Output ---")

# 1. Create model
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

#2. Fit UNSCALED data
dt_clf.fit(X_train, y_train)

# 3. Predict test set
y_pred_dt = dt_clf.predict(X_test)

# 4.Output results
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

"""
COMMENT 1: Comparing Decision Tree accuracy to KNN
For the Iris dataset, the Decision Tree accuracy is generally similar to 
KNN (often identical depending on the random split). Both models easily 
handle the distinct setosa class and perform good on the remaining two.

COMMENT 2: Does scaled vs unscaled data affect Decision Trees?
No, scaling makes no difference 
a Decision Tree just looks 
for threshold splits. If we scale the data, 
the tree will just make the same split using the new scaled number . The resulting tree is identical.
"""


# Logistic Regression and Regularization - ommision corrected

print("\n--- Logistic Regression and Regularization ---")

C_values = [0.01, 1.0, 100]

for c in C_values:
    log_reg = LogisticRegression(C=c, max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    coef_magnitude = np.abs(log_reg.coef_).sum()
    print(f"C={c:5.2f} | Total Coefficient Magnitude: {coef_magnitude:.4f}")

"""
COMMENT: How does total coefficient magnitude change with regularization?
In scikit-learn, C is the inverse of regularisation strength therefore smaller C = stronger penalty. 
As C decreases, the total magnitude of the coefficients 
shrinks a lot. The model is penalized for having large weights, which forces it 
to shrink less important feature weights down toward zero, reducing overfitting.
"""

# Principal Component Analysis (PCA)

print("\n--- PCA Setup ---")

# Load digits dataset
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Q1
print("\n--- PCA Q1 Output ---")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(12, 3))

# Loop 0 through 9
for i in range(10):
    # Find the index of digit 'i'
    first_idx = np.where(y_digits == i)[0][0]
    axes[i].imshow(images[first_idx], cmap='gray_r')
    axes[i].set_title(str(i))
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('outputs/sample_digits.png')
plt.clf()
print("Saved outputs/sample_digits.png")


#Q2: PCA 2D Projection
print("\n--- PCA Q2 Output ---")

# Fit PCA and transform
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')

plt.title('PCA 2D Projection of Digits')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('outputs/pca_2d_projection.png')
plt.clf()
print("Saved outputs/pca_2d_projection.png")

"""
COMMENT: Do same-digit images tend to cluster together in this 2D space?
i think so. Because we compressed 64 dimensions 
(pixels) down to just 2, there is definitely some overlap in the middle 
where similar-looking numbers get jumbled together?
"""


#Q3: Cumulative Explained Variance
print("\n--- PCA Q3 Output ---")

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='.', linestyle='-', color='b')
plt.axhline(y=0.80, color='r', linestyle='--', label="80% Threshold")

plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.legend()
plt.savefig('outputs/pca_variance_explained.png')
plt.clf()
print("Saved outputs/pca_variance_explained.png")

"""
COMMENT: Approximately how many components do you need to explain 80% of the variance?
Looking at where the blue line crosses the red 80% threshold, it takes 
approximately 13 components to capture 80% of the variance in the dataset.
"""


#Q4: Digit Reconstruction
print("\n--- PCA Q4 Output ---")

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]

# Create a grid
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

# 1. Plot the "Original" row at the top
for j in range(5):
    axes[0, j].imshow(images[j], cmap='gray_r')
    if j == 0:
        axes[0, j].set_ylabel("Original", size='large', weight='bold')
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])

# 2. Plot the reconstructions for 'n'
for row_idx, n in enumerate(n_values):
    for col_idx in range(5):
        ax = axes[row_idx + 1, col_idx]
        recon = reconstruct_digit(col_idx, scores, pca, n)
        
        ax.imshow(recon, cmap='gray_r')
        if col_idx == 0:
            ax.set_ylabel(f"n = {n}", size='large')
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("PCA Reconstructions of the First 5 Digits")
plt.tight_layout()
plt.savefig('outputs/pca_reconstructions.png')
plt.clf()
print("Saved outputs/pca_reconstructions.png")

"""
COMMENT: At what n do the digits become clearly recognizable, and does that match the curve?
The digits around n=15. This matches  
variance curve from Q3, which showed that ~13-15 components capture over 80% of 
the underlying information. By n=40, the reconstruction is almost 
the original, even though we are still missing 24 dimensions of data!
"""