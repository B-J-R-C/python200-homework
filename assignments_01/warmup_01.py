"""
Python 200: Assignment 01
Author: Ben Chapman
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Pandas ---

# Pandas Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("--- Pandas Q1 Output ---")

# Print the first three rows
print(f"First three rows:\n{df.head(3)}\n")

# Print the shape
print(f"DataFrame Shape: {df.shape}")
print(f"Num Rows: {len(df)}")
print(f"Num Columns: {len(df.columns)}\n")

# Print the data types
print(f"Data Types:\n{df.dtypes}\n")

# Pandas Q2
print("--- Pandas Q2 Output ---")

# Filter for passed == True AND grade > 80
filtered_df = df[(df['passed'] == True) & (df['grade'] > 80)]

print(f"Students who passed with a grade > 80:\n{filtered_df}\n")

# Pandas Q3
print("--- Pandas Q3 Output ---")

# Create a new column by adding 5 to the existing 'grade' column
df['grade_curved'] = df['grade'] + 5

print(f"DataFrame with curved grades:\n{df}\n")

# Pandas Q4
print("--- Pandas Q4 Output ---")

# Create the 'name_upper' column using the .str accessor
df['name_upper'] = df['name'].str.upper()

# Print only the 'name' and 'name_upper' columns
print(f"Names and Uppercase Names:\n{df[['name', 'name_upper']]}\n")

# Pandas Q5
print("--- Pandas Q5 Output ---")

# Group by city and calculate the mean of the 'grade' column
city_avg_grades = df.groupby('city')['grade'].mean()

print(f"Average grades by city:\n{city_avg_grades}\n")

# Pandas Q7
print("--- Pandas Q7 Output ---")

# Sort by 'grade' descending, then grab the top 3 rows
top_3_grades = df.sort_values(by='grade', ascending=False).head(3)

print(f"Top 3 Students by Grade:\n{top_3_grades}\n")


# --- NumPy ---

# NumPy Q1
print("--- NumPy Q1 Output ---")
arr_1d = np.array([10, 20, 30, 40, 50])

print(f"1D Array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")
print(f"Data Type (dtype): {arr_1d.dtype}")
print(f"Number of Dimensions (ndim): {arr_1d.ndim}\n")

# NumPy Q2
print("--- NumPy Q2 Output ---")
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"2D Array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")
print(f"Size (total elements): {arr_2d.size}\n")

# NumPy Q3
print("--- NumPy Q3 Output ---")
top_left_2x2 = arr_2d[:2, :2]

print(f"Top-left 2x2 block:\n{top_left_2x2}\n")

# NumPy Q4
print("--- NumPy Q4 Output ---")
# Built-in functions
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))

print(f"3x4 Array of Zeros:\n{zeros_arr}\n")
print(f"2x5 Array of Ones:\n{ones_arr}\n")

# NumPy Q5
print("--- NumPy Q5 Output ---")
# arange works
range_arr = np.arange(0, 50, 5)

print(f"Array (np.arange(0, 50, 5)):\n{range_arr}")
print(f"Shape: {range_arr.shape}")
print(f"Mean: {range_arr.mean()}")
print(f"Sum: {range_arr.sum()}")
# Roundingstd deviation
print(f"Standard Deviation: {range_arr.std():.2f}\n")

# NumPy Q6
print("--- NumPy Q6 Output ---")
# loc = mean, scale = standard deviation
rand_norm = np.random.normal(loc=0.0, scale=1.0, size=200)

# randomly generated, the mean will be close to 0
print(f"Generated 200 random values.")
print(f"Calculated Mean (approx 0): {rand_norm.mean():.4f}")
print(f"Calculated Std Dev (approx 1): {rand_norm.std():.4f}\n")

# --- Matplotlib ---

# Matplotlib Q1
print("--- Matplotlib Q1: Saving line plot to outputs/q1_squares.png ---")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Create the plot
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

# Save and clear the figure
plt.savefig("outputs/q1_squares.png")
plt.clf() 

# Matplotlib Q2
print("--- Matplotlib Q2: Saving bar plot to outputs/q2_subjects.png ---")
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# Create the bar plot
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

# Save and clear the figure
plt.savefig("outputs/q2_subjects.png")
plt.clf()

# Matplotlib Q3
print("--- Matplotlib Q3: Saving scatter plot to outputs/q3_scatter.png ---")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# Create the scatter plots (calling plt.scatter twice overlays them on the same figure)
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()

# Save and clear the figure
plt.savefig("outputs/q3_scatter.png")
plt.clf()

# Matplotlib Q4
print("--- Matplotlib Q4: Saving subplots to outputs/q4_subplots.png ---")

# Create a figure and an array of axes (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot (Q1 data)
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Right subplot (Q2 data)
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

# Automatically adjust padding so labels don't overlap
plt.tight_layout()

# Save and clear
plt.savefig("outputs/q4_subplots.png")
plt.clf()

# --- Descriptive Statistics ---

# Descriptive Stats Q1
print("--- Descriptive Stats Q1 Output ---")
data_stats = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]


print(f"Mean: {np.mean(data_stats)}")
print(f"Median: {np.median(data_stats)}")
print(f"Variance: {np.var(data_stats):.2f}")
print(f"Standard Deviation: {np.std(data_stats):.2f}\n")


# Descriptive Stats Q2
print("--- Descriptive Stats Q2: Saving histogram to outputs/stats_q2_hist.png ---")
scores = np.random.normal(65, 10, 500)


plt.hist(scores, bins=20, edgecolor='black')
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.savefig("outputs/stats_q2_hist.png")
plt.clf()


# Descriptive Stats Q3
print("--- Descriptive Stats Q3: Saving boxplot to outputs/stats_q3_box.png ---")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")

plt.savefig("outputs/stats_q3_box.png")
plt.clf()


# Descriptive Stats Q4
print("--- Descriptive Stats Q4: Saving comparison boxplot to outputs/stats_q4_box.png ---")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")

plt.savefig("outputs/stats_q4_box.png")
plt.clf()

# --- Q4 COMMENTARY ---
# The exponential distribution is heavily right-skewed 
# For the skewed data, the median is a better measure of central tendency because it is resistant to being pulled upward by those extreme outlier values in the tail.


# Descriptive Stats Q5
import statistics

print("--- Descriptive Stats Q5 Output ---")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data 1:")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {statistics.mode(data1)}\n")

print("Data 2:")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {statistics.mode(data2)}\n")

# --- Q5 COMMENTARY ---
# The mean and median are so different for data2 because of the extreme outlier (150). 
# The mean is calculated by summing all values and dividing by the total count
# --- Hypothesis Testing ---
from scipy import stats

# Hypothesis Q1
print("--- Hypothesis Q1 Output ---")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Run independent samples t-test
t_stat_1, p_val_1 = stats.ttest_ind(group_a, group_b)

print(f"Independent t-statistic: {t_stat_1:.4f}")
print(f"p-value: {p_val_1:.6f}\n")


# Hypothesis Q2
print("--- Hypothesis Q2 Output ---")
alpha = 0.05

if p_val_1 < alpha:
    print(f"Statistically significant (p < {alpha}). We reject the null hypothesis.\n")
else:
    print(f"Not statistically significant (p >= {alpha}). We fail to reject the null hypothesis.\n")


# Hypothesis Q3
print("--- Hypothesis Q3 Output ---")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Run paired (related) samples t-test
t_stat_3, p_val_3 = stats.ttest_rel(before, after)

print(f"Paired t-statistic: {t_stat_3:.4f}")
print(f"p-value: {p_val_3:.6f}\n")


# Hypothesis Q4
print("--- Hypothesis Q4 Output ---")
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

# Run one-sample t-test
t_stat_4, p_val_4 = stats.ttest_1samp(scores, benchmark)

print(f"One-sample t-statistic: {t_stat_4:.4f}")
print(f"p-value: {p_val_4:.4f}\n")


# Hypothesis Q5
print("--- Hypothesis Q5 Output ---")
# Re-running Q1
t_stat_5, p_val_5 = stats.ttest_ind(group_a, group_b, alternative='less')

print(f"One-tailed p-value (group_a < group_b): {p_val_5:.6f}\n")


# Hypothesis Q6
print("--- Hypothesis Q6 Output ---")
# Plain-language
print("The analysis shows that Group B scored significantly higher than Group A on average. Because the p-value is extremely low (well below our 0.05 threshold), we can confidently conclude that this difference is a true effect rather than a result of random chance.\n")

# --- Correlation ---
import seaborn as sns

# Correlation Q1
print("--- Correlation Q1 Output ---")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate correlation matrix
corr_matrix_q1 = np.corrcoef(x, y)

print(f"Full Correlation Matrix:\n{corr_matrix_q1}")
print(f"Correlation Coefficient (x,y): {corr_matrix_q1[0, 1]}\n")

# --- Q1 COMMENTARY ---
# I expect the correlation to be exactly 1.0. 
# Because 'y' is simply 2 times 'x' for every data point, there is a perfect positive linear relationship between the two variables.


# Correlation Q2
print("--- Correlation Q2 Output ---")
from scipy.stats import pearsonr

x_q2 = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y_q2 = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# Calculate Pearson correlation and p-value
corr_coef_q2, p_val_q2 = pearsonr(x_q2, y_q2)

print(f"Correlation Coefficient: {corr_coef_q2:.4f}")
print(f"p-value: {p_val_q2:.6f}\n")


# Correlation Q3
print("--- Correlation Q3 Output ---")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df_people = pd.DataFrame(people)

# Compute correlation matrix on the dataframe
df_corr = df_people.corr()

print(f"DataFrame Correlation Matrix:\n{df_corr}\n")


# Correlation Q4
print("--- Correlation Q4: Saving scatter plot to outputs/corr_q4_scatter.png ---")
x_q4 = [10, 20, 30, 40, 50]
y_q4 = [90, 75, 60, 45, 30]

plt.scatter(x_q4, y_q4)
plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")

# Save and clear the figure
plt.savefig("outputs/corr_q4_scatter.png")
plt.clf()


# Correlation Q5
print("--- Correlation Q5: Saving heatmap to outputs/corr_q5_heatmap.png ---")
# Generate heatmap using the correlation matrix from Q3
sns.heatmap(df_corr, annot=True)
plt.title("Correlation Heatmap")

# Automatically adjust layout so labels aren't cut off
plt.tight_layout()

# Save and clear the figure
plt.savefig("outputs/corr_q5_heatmap.png")
plt.clf()

# --- Pipelines ---

# Pipeline Q1
print("--- Pipeline Q1 Output ---")
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    """Takes a NumPy array and returns a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

def clean_data(series):
    """Removes NaN values from the Series."""
    return series.dropna()

def summarize_data(series):
    """Returns a dictionary of summary statistics for the cleaned Series."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    """Chains the processing steps together into a single pipeline."""
    # Step 1: Ingest and format
    raw_series = create_series(arr)
    
    # Step 2: Clean and transform
    cleaned_series = clean_data(raw_series)
    
    # Step 3: Analyze and output
    summary_dict = summarize_data(cleaned_series)
    
    return summary_dict

# Run the pipeline and capture the result
final_summary = data_pipeline(arr)

# Print each key and value from the dictionary
print("Pipeline Results:")
for key, value in final_summary.items():
    print(f"{key.capitalize()}: {value:.4f}")
print("\n")