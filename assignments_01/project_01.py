"""
Python 200: Project 01 - World Happiness Pipeline
Author: [Your Name]
"""

import pandas as pd
import os
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- TASK 1 ---

@task(name="Load Happiness Data", retries=3, retry_delay_seconds=2)
def load_multiple_years(data_dir: str, output_path: str):
    """Loads multiple years of CSV data, adds a year column, and merges them."""
    logger = get_run_logger()
    logger.info(f"Looking for data in: {data_dir}")
    
    all_years_data = []
    
    for year in range(2015, 2025):
        file_path = os.path.join(data_dir, f"world_happiness_{year}.csv")
        
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',')
            df['year'] = year
            
            # 1. Standardize columns before
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '_')
            
            # 2. find the region column dynamically and force it to be  'region'
            for col in df.columns:
                if 'region' in col:
                    df.rename(columns={col: 'region'}, inplace=True)
                    break # Stop once found
            
            all_years_data.append(df)
            logger.info(f"Successfully loaded data for {year} ({len(df)} rows)")
            
        except FileNotFoundError:
            logger.error(f"Could not find file: {file_path}")
            raise
            
    
    merged_df = pd.concat(all_years_data, ignore_index=True)
    logger.info(f"Merged dataset complete. Total rows: {len(merged_df)}")
    
    # Save merged result
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset to {output_path}")
    
    return merged_df

# --- TASK 2: Descriptive Statistics ---

@task(name="Calculate Descriptive Statistics")
def calculate_descriptive_stats(df):
    """Computes and logs descriptive statistics for the happiness dataset."""
    logger = get_run_logger()
    
    # Overall stats
    logger.info("--- Overall Happiness Statistics ---")
    mean_score = df['happiness_score'].mean()
    median_score = df['happiness_score'].median()
    std_score = df['happiness_score'].std()
    
    logger.info(f"Mean: {mean_score:.4f}")
    logger.info(f"Median: {median_score:.4f}")
    logger.info(f"Standard Deviation: {std_score:.4f}")
    
    # Group by Year
    logger.info("--- Mean Happiness by Year ---")
    yearly_avg = df.groupby('year')['happiness_score'].mean()
    # Using .to_string() formats the Pandas Series nicely for the logger
    logger.info(f"\n{yearly_avg.to_string()}")
    
    # Group by Region
    logger.info("--- Mean Happiness by Region ---")
    # Sort descending
    regional_avg = df.groupby('region')['happiness_score'].mean().sort_values(ascending=False)
    logger.info(f"\n{regional_avg.to_string()}")

    # --- TASK 3: Visual Exploration ---

@task(name="Generate Visualizations")
def create_visualizations(df):
    """Generates and saves exploratory data visualizations."""
    logger = get_run_logger()
    logger.info("Starting visualization generation...")

    # 1. Histogram of Happiness score
    plt.figure(figsize=(8, 6))
    sns.histplot(df['happiness_score'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of All Happiness Scores (2015-2024)')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    plt.savefig('outputs/happiness_histogram.png')
    plt.clf()
    logger.info("Saved happiness_histogram.png")

    # 2. Boxplot across jears
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='year', y='happiness_score', data=df, palette='Set3')
    plt.title('Happiness Score Distributions by Year')
    plt.xlabel('Year')
    plt.ylabel('Happiness Score')
    plt.savefig('outputs/happiness_by_year.png')
    plt.clf()
    logger.info("Saved happiness_by_year.png")

    # 3. Scatter: GDP vs Happiness

    gdp_col = [col for col in df.columns if 'gdp' in col.lower() or 'economy' in col.lower()]
    
    if gdp_col:
        actual_gdp_col = gdp_col[0]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual_gdp_col, y='happiness_score', data=df, alpha=0.7)
        plt.title('GDP per Capita vs Happiness Score')
        plt.xlabel(actual_gdp_col)
        plt.ylabel('Happiness Score')
        plt.savefig('outputs/gdp_vs_happiness.png')
        plt.clf()
        logger.info(f"Saved gdp_vs_happiness.png (using column: {actual_gdp_col})")
    else:
        logger.warning("Could not find a GDP column to generate the scatter plot!")

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Filter to only numeric columns so the correlation math don't break
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout() # stop labels be cut off
    plt.savefig('outputs/correlation_heatmap.png')
    plt.clf()
    logger.info("Saved correlation_heatmap.png")

    # --- TASK 4: Hypothesis Testing ---

@task(name="Hypothesis Testing")
def run_hypothesis_tests(df):
    """Runs statistical tests on the happiness dataset and logs the results."""
    logger = get_run_logger()
    logger.info("Starting Hypothesis Testing...")

    # --- Test 1: 2019 vs 2020 (Pandemic Impact) ---
    logger.info("--- Test 1: Global Happiness (2019 vs 2020) ---")
    
    # Isolate the scores for each year, dropping NaNs
    scores_2019 = df[df['year'] == 2019]['happiness_score'].dropna()
    scores_2020 = df[df['year'] == 2020]['happiness_score'].dropna()
    
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    
    # independent samples t-test
    t_stat_1, p_val_1 = stats.ttest_ind(scores_2019, scores_2020)
    
    logger.info(f"2019 Mean Happiness: {mean_2019:.4f}")
    logger.info(f"2020 Mean Happiness: {mean_2020:.4f}")
    logger.info(f"T-statistic: {t_stat_1:.4f}")
    logger.info(f"p-value: {p_val_1:.4f}")
    
    alpha = 0.05
    if p_val_1 < alpha:
        logger.info(
            f"Interpretation: The p-value ({p_val_1:.4f}) is less than our alpha of {alpha}. "
            "This indicates a statistically significant difference in global happiness between 2019 and 2020. "
            "The data suggests the onset of the pandemic measurably altered global happiness scores."
        )
    else:
        logger.info(
            f"Interpretation: The p-value ({p_val_1:.4f}) is greater than our alpha of {alpha}. "
            "We fail to reject the null hypothesis. Despite the onset of the pandemic, global "
            "average happiness scores did not change by a statistically significant amount."
        )

    # --- Test 2: Regional Comparison ---
    # Dynamically find highest and lowest regions
    regional_means = df.groupby('region')['happiness_score'].mean().sort_values(ascending=False)
    top_region = regional_means.index[0]
    bottom_region = regional_means.index[-1]
    
    logger.info(f"\n--- Test 2: Regional Comparison ({top_region} vs {bottom_region}) ---")
    
    scores_top = df[df['region'] == top_region]['happiness_score'].dropna()
    scores_bottom = df[df['region'] == bottom_region]['happiness_score'].dropna()
    
    t_stat_2, p_val_2 = stats.ttest_ind(scores_top, scores_bottom)
    
    logger.info(f"{top_region} Mean: {scores_top.mean():.4f}")
    logger.info(f"{bottom_region} Mean: {scores_bottom.mean():.4f}")
    logger.info(f"T-statistic: {t_stat_2:.4f}")
    logger.info(f"p-value: {p_val_2:.10f}") # Using more decimal places since regional p-values are often tiny
    
    if p_val_2 < alpha:
        logger.info(
            f"Interpretation: The p-value is extremely low. There is a massive, statistically "
            f"significant gap in happiness between {top_region} and {bottom_region}, confirming "
            f"that geopolitical and economic regions play a profound role in life evaluation."
        )
    else:
        logger.info("Interpretation: No statistically significant difference found.")

        # --- TASK 5: Correlation and Multiple Comparisons ---

@task(name="Correlation Analysis")
def run_correlation_analysis(df):
    """Computes correlations, applies Bonferroni correction, and logs significance."""
    logger = get_run_logger()
    logger.info("Starting Correlation Analysis and Multiple Comparisons...")

    # Filter to numeric columns, excluding the target itself
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    explanatory_vars = [col for col in numeric_cols if col not in ['happiness_score', 'year']]

    results = []

    logger.info("--- Pearson Correlations with Happiness Score ---")
    for col in explanatory_vars:
        # pearsonr fails if  NaNs, so drop 4 specific pair
        valid_data = df[['happiness_score', col]].dropna()
        
        if len(valid_data) > 1: # Need at least 2 points to correlate
            r_val, p_val = stats.pearsonr(valid_data['happiness_score'], valid_data[col])
            results.append((col, r_val, p_val))
            # Scientific notation for p-values
            logger.info(f"{col}: r = {r_val:.4f}, p-value = {p_val:.6e}")

    # Calculate Bonferroni Correction
    num_tests = len(results)
    original_alpha = 0.05
    adjusted_alpha = original_alpha / num_tests if num_tests > 0 else 0.05

    logger.info("\n--- Significance Checks (Original vs Bonferroni) ---")
    logger.info(f"Number of tests run: {num_tests}")
    logger.info(f"Original Alpha: {original_alpha}")
    logger.info(f"Adjusted Alpha (Bonferroni): {adjusted_alpha:.6f}\n")

    for col, r_val, p_val in results:
        sig_original = p_val < original_alpha
        sig_adjusted = p_val < adjusted_alpha

        if sig_adjusted:
            status = "Significant at BOTH (Original & Adjusted)"
        elif sig_original:
            status = "Significant ONLY at Original Alpha (Failed Bonferroni)"
        else:
            status = "Not Significant"

        logger.info(f"{col}: {status}")

        # --- TASK 6: Summary Report ---

@task(name="Generate Summary Report")
def generate_summary_report(df):
    """Generates a human-readable summary of the key findings."""
    logger = get_run_logger()
    
    logger.info("==========================================")
    logger.info("      WORLD HAPPINESS PIPELINE REPORT     ")
    logger.info("==========================================")

    # 1. Dataset Scope
    # Dynamically find the country column
    country_col = [col for col in df.columns if 'country' in col.lower()][0]
    num_countries = df[country_col].nunique()
    num_years = df['year'].nunique()
    
    logger.info(f"1. DATASET SCOPE: Analyzed {num_countries} unique countries over {num_years} years.")

    # 2. Regional Rankings
    regional_means = df.groupby('region')['happiness_score'].mean().sort_values(ascending=False)
    top_3 = ", ".join(regional_means.head(3).index.tolist())
    bottom_3 = ", ".join(regional_means.tail(3).index.tolist())
    
    logger.info(f"2. REGIONAL RANKINGS:")
    logger.info(f"   - Top 3 Regions: {top_3}")
    logger.info(f"   - Bottom 3 Regions: {bottom_3}")

    # 3. Pandemic Impact (2019 vs 2020)
    scores_2019 = df[df['year'] == 2019]['happiness_score'].dropna()
    scores_2020 = df[df['year'] == 2020]['happiness_score'].dropna()
    _, p_val_pandemic = stats.ttest_ind(scores_2019, scores_2020)
    
    if p_val_pandemic < 0.05:
        pandemic_result = "Statistically significant change detected in global happiness."
    else:
        pandemic_result = "No statistically significant change detected at the global level."
        
    logger.info(f"3. PANDEMIC IMPACT (2019 vs 2020): {pandemic_result}")

    # 4. Strongest Correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    explanatory_vars = [col for col in numeric_cols if col not in ['happiness_score', 'year']]
    
    best_var = None
    max_r = 0
    adjusted_alpha = 0.05 / len(explanatory_vars) if len(explanatory_vars) > 0 else 0.05
    
    for col in explanatory_vars:
        valid_data = df[['happiness_score', col]].dropna()
        if len(valid_data) > 1:
            r_val, p_val = stats.pearsonr(valid_data['happiness_score'], valid_data[col])
            # Check if it passes Bonferroni AND is the strongest relationship we've seen so far
            if p_val < adjusted_alpha and abs(r_val) > abs(max_r):
                max_r = r_val
                best_var = col
                
    logger.info(f"4. STRONGEST DRIVER: '{best_var}' had the strongest correlation with happiness (r = {max_r:.4f}), passing the strict Bonferroni threshold.")
    logger.info("==========================================\n")


# --- MAIN PIPELINE FLOW ---

@flow(name="World Happiness Analysis Flow")
def happiness_pipeline():
    """Main execution flow for the World Happiness analysis."""
    logger = get_run_logger()
    logger.info("Starting the World Happiness Pipeline...")
    
    # Define file paths
    data_dir = "../assignments/resources/happiness_project"
    output_csv = "outputs/merged_happiness.csv"
    
    # Execute Task 1: Load and Merge
    merged_data = load_multiple_years(data_dir, output_csv)
    
    # Execute Task 2: Descriptive Statistics
    calculate_descriptive_stats(merged_data)
    
    # Execute Task 3: Visual Exploration
    create_visualizations(merged_data)
    
    # Execute Task 4: Hypothesis Testing
    run_hypothesis_tests(merged_data)
    
    # Execute Task 5: Correlation Analysis
    run_correlation_analysis(merged_data)
    
    # Execute Task 6: Summary Report
    generate_summary_report(merged_data)

if __name__ == "__main__":
    happiness_pipeline()