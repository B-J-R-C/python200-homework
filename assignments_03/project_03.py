"""
Python 200: Assignment 03 - Spam or Ham? A Classifier Shootout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


# TASK 1: Load and Explore

print("--- TASK 1: Load and Explore ---")

# The Spambase dataset no column headers, so we defined manually as could tnget to work otherwise
column_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "spam_label"
]

# Load the dataset header=None
df = pd.read_csv('spambase.csv', header=None, names=column_names)

print(f"Total emails in dataset: {df.shape[0]}")
print("\nClass Balance:")
print(df['spam_label'].value_counts(normalize=True) * 100)


features_to_plot = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']

for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    # log scale
    sns.boxplot(x='spam_label', y=feature, data=df, palette='Set2')
    plt.yscale('symlog')
    plt.title(f'Distribution of {feature} (Spam vs Ham)')
    plt.savefig(f'outputs/boxplot_{feature}.png')
    plt.clf()

print("\nSaved boxplots to outputs/")

"""
COMMENT: Feature Distributions & Scales
1. Differences: The differences are big- esp for exclamation marks 
   and capital run lengths. Spam emails have higher distributions of these features.
2. Skew toward zero. Most emails don't use these specific words. 
3. Varying Scales: Frequencies are tiny decimals (0 to 1), but capital run lengths 
   are huge integers (into the thousands). This matters immensely for distance-based 
   models like KNN or Logistic Regression—if we don't scale the data, the 'capital_run_length' 
   feature will completely dominate the math just because its raw numbers are bigger.
"""


# TASK 2: Prepare Data

print("\n--- TASK 2: Prepare Your Data ---")

X = df.drop('spam_label', axis=1)
y = df['spam_label']

# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. PCA Preprocessing
pca_full = PCA()
pca_full.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find 'n' where cumulative variance reaches 90%
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1 
print(f"Number of components to reach 90% variance: {n_components_90}")

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='.')
plt.axhline(y=0.90, color='r', linestyle='--', label="90% Threshold")
plt.axvline(x=n_components_90, color='g', linestyle=':', label=f"n={n_components_90}")
plt.title('PCA Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.legend()
plt.savefig('outputs/spambase_pca_variance.png')
plt.clf()

# 4. Create final PCA array using 'n'
pca_optimal = PCA(n_components=n_components_90)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_test_pca = pca_optimal.transform(X_test_scaled)



# TASK 3: A Classifier Comparison

print("\n--- TASK 3: Classifier Comparison ---")

#KNN
knn_unscaled = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print(f"KNN (Unscaled) Accuracy: {accuracy_score(y_test, knn_unscaled.predict(X_test)):.4f}")

knn_scaled = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
print(f"KNN (Scaled) Accuracy:   {accuracy_score(y_test, knn_scaled.predict(X_test_scaled)):.4f}")

knn_pca = KNeighborsClassifier(n_neighbors=5).fit(X_train_pca, y_train)
print(f"KNN (PCA) Accuracy:      {accuracy_score(y_test, knn_pca.predict(X_test_pca)):.4f}")

#Decision Trees (Depth Experiment
print("\nDecision Tree Depth Experiment:")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"  Depth: {str(depth):4s} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

"""
COMMENT: Decision Tree Depth
As depth increases, Train Accuracy hits 1.000 (100%), but Test Accuracy peaked around 
depth 10 and then drops. Overfitting: tree memorized the training 
data (1 leaf per sample) but lost ability to generalize to new data.
I would use max_depth=10 in production because it provides the highest test accuracy 
without fully memorizing the training set.
"""

best_dt = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train)
print(f"\nDecision Tree (Depth 10) Classification Report:")
print(classification_report(y_test, best_dt.predict(X_test)))

# --- Random ---
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# --- Regression ---
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear').fit(X_train_scaled, y_train)
print(f"Logistic Regression (Scaled) Accuracy: {accuracy_score(y_test, log_reg_scaled.predict(X_test_scaled)):.4f}")

log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear').fit(X_train_pca, y_train)
print(f"Logistic Regression (PCA) Accuracy:    {accuracy_score(y_test, log_reg_pca.predict(X_test_pca)):.4f}")


"""
COMMENT: Classifier Comparison Summary
1. Best Model: The Random Forest performs the best overall.
2. PCA vs Non-PCA: For Logistic Regression and KNN, the Scaled data performed slightly 
   better than the PCA data. Because dropped 10% of the variance to compress the 
   data, lost a tiny bit of predictive power. PCA is great for speed and memory, 
   but sometimes sacrifices a fraction of accuracy.
3. The Right Metric: For a spam filter, False Positives (labeling a real, important 
   email as spam) are catastrophic. Missing an interview request is much worse than 
   seeing a random spam email in your inbox (False Negative). Therefore, we should 
   optimize for HIGH PRECISION on the Spam class (minimizing false positives) rather 
   than raw accuracy.
"""

# Confusion Matrix
cm = confusion_matrix(y_test, rf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.savefig('outputs/best_model_confusion_matrix.png')
plt.clf()

"""
COMMENT: Error Types
Looking at the confusion matrix, the model has very few False Positives, which aligns perfectly with our goal of not sending 
legitimate emails to the spam folder.
"""

# Feature Importances (Tree vs Forest)
print("\nTop 10 Features (Decision Tree):")
dt_importances = pd.Series(best_dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print(dt_importances.head(10))

print("\nTop 10 Features (Random Forest):")
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(rf_importances.head(10))

# Save Bar Chart
plt.figure(figsize=(10, 6))
rf_importances.head(15).plot(kind='barh', color='teal').invert_yaxis()
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importances.png')
plt.clf()



# ==========================================
# TASK 4: Cross-Validation
# ==========================================
print("\n--- TASK 4: Cross-Validation ---")

# We use a list of tuples: (Name, Model, X_data_to_use)
models_to_cv = [
    ("KNN (Unscaled)", KNeighborsClassifier(n_neighbors=5), X_train),
    ("KNN (Scaled)", KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    ("KNN (PCA)", KNeighborsClassifier(n_neighbors=5), X_train_pca),
    ("Decision Tree (d=10)", DecisionTreeClassifier(max_depth=10, random_state=42), X_train),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    ("Logistic Reg (Scaled)", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_scaled),
    ("Logistic Reg (PCA)", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'), X_train_pca)
]

for name, model, x_data in models_to_cv:
    scores = cross_val_score(model, x_data, y_train, cv=5)
    print(f"{name:25s} | Mean CV Acc: {scores.mean():.4f} | Std Dev: {scores.std():.4f}")

"""
COMMENT: CV Results
Random Forest is still most accurate (highest mean) and most stable (lowest standard 
deviation) across the board. Now that we tested all models properly, we can clearly see 
the massive jump in performance KNN gets when moving from Unscaled to Scaled. 
The ranking holds true: Random Forest reigns supreme, LogReg is second, and Unscaled 
KNN is severely hindered by the different feature magnitudes.
"""



# TASK 5: Building a Prediction Pipeline

print("\n--- TASK 5: Building Pipelines ---")

# Pipeline 1: Best Tree-Based Model (Random Forest)

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

tree_pipeline.fit(X_train, y_train)
print("\n--- Random Forest Pipeline Classification Report ---")
print(classification_report(y_test, tree_pipeline.predict(X_test)))


# Pipeline 2: Best Non-Tree Model (Logistic Regression)

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

linear_pipeline.fit(X_train, y_train)
print("\n--- Logistic Regression Pipeline Classification Report ---")
print(classification_report(y_test, linear_pipeline.predict(X_test)))

"""
COMMENT: Pipeline Structures & Value
1. Structure: They have different structures. The Random Forest pipeline just passes 
   the raw data straight to the classifier because trees are scale-invariant. 
2. Practical Value: Pipelines are good for deployment. Without a pipeline, if 
   a software engineer wanted to integrate my model into an email app, they would have 
   to write custom code to load my exact scaler, scale the new email, load my exact 
   PCA object, transform the email, then predict. With a pipeline, they just call 
   `pipeline.predict(new_email)` and all the preprocessing happens invisibly and perfectly.
"""