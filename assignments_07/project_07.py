
import os
# Force a non-interactive matplotlib backend BEFORE anything imports pyplot,
# so the agent's plotting code can save PNGs in a headless environment.
os.environ.setdefault("MPLBACKEND", "Agg")
 
import re
import glob
import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv
 
from smolagents import CodeAgent, OpenAIServerModel, tool
 
# Load environment variables (make sure your .env has your OpenAI key!)
load_dotenv()
 
# Global DataFrame holding the merged World Happiness data. This persists across
# agent.run() calls regardless of the `reset` flag, because the tools close over
# module state - so once loaded, the data stays available for every query.
df = None
 
# ==========================================================================
# --- Data source configuration ---
# ==========================================================================
# Put the yearly CSVs in this folder. Filenames should contain the year, e.g.
# 2018.csv or world-happiness-2018.csv (a 4-digit year is pulled from the name).
LOCAL_DIR = "happiness_data"
 
# Optional fallback URLs, used only if LOCAL_DIR has no CSVs. Leave empty ({}).
REMOTE_URLS = {
    # 2018: "https://raw.githubusercontent.com/.../2018.csv",
}
 
# Canonical names mapped to the *normalized* form of every variant we've seen.
# Normalizing (lowercase + strip non-alphanumerics) collapses punctuation/spacing
# so "Regional indicator", "Region", "Economy (GDP per Capita)" and the R-style
# "Economy..GDP.per.Capita." all map without listing every exact spelling.
COLUMN_ALIASES = {
    "country":         {"country", "countryname", "countryorregion"},
    "region":          {"region", "regionalindicator"},
    "happiness_score": {"happinessscore", "score", "ladderscore"},
    "gdp_per_capita":  {"economygdppercapita", "gdppercapita", "loggedgdppercapita"},
    "life_expectancy": {"healthlifeexpectancy", "healthylifeexpectancy"},
}
 
# Canonical numeric columns we coerce so stats never trip over stray strings.
NUMERIC_COLS = ["happiness_score", "gdp_per_capita", "life_expectancy"]
 
 
def _norm(name: str) -> str:
    """Lowercase a column name and strip everything that isn't a letter/digit."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
 
 
def _year_from_name(path: str):
    """Pull a 4-digit year (19xx/20xx) from a filename, or None if absent."""
    match = re.search(r"(?:19|20)\d{2}", os.path.basename(path))
    return int(match.group()) if match else None
 
 
def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename ONE frame's columns to canonical names. Applied per frame, before
    concat, so we never end up with duplicate column labels after merging."""
    rename_map = {}
    for col in frame.columns:
        key = _norm(col)
        for canonical, variants in COLUMN_ALIASES.items():
            if key in variants:
                rename_map[col] = canonical
                break
    return frame.rename(columns=rename_map)
 
 
def _read_csv_smart(src) -> pd.DataFrame:
    """Read a CSV while auto-detecting the delimiter. Handles comma- AND
    semicolon-delimited files (and tab/pipe), with a BOM-tolerant encoding."""
    # sep=None + the python engine asks pandas to sniff the delimiter.
    frame = pd.read_csv(src, sep=None, engine="python",
                        on_bad_lines="skip", encoding="utf-8-sig")
    # Safety net: if everything collapsed into a single column, the sniff failed
    # (e.g. a ;-delimited file read as comma). Detect the delimiter from the
    # header string we got and re-read explicitly.
    if frame.shape[1] == 1:
        header = str(frame.columns[0])
        for delim in (";", "\t", "|"):
            if delim in header:
                frame = pd.read_csv(src, sep=delim, engine="python",
                                    on_bad_lines="skip", encoding="utf-8-sig")
                break
    return frame
 
 
def _candidate_sources():
    """Return [(year, path_or_url), ...] and a label, preferring local files."""
    local = sorted(glob.glob(os.path.join(LOCAL_DIR, "*.csv")))
    if local:
        return [(_year_from_name(p), p) for p in local], f"local folder '{LOCAL_DIR}'"
    if REMOTE_URLS:
        return [(int(y), url) for y, url in REMOTE_URLS.items()], "GitHub URLs"
    return [], None
 
 
# ==========================================================================
# --- Task 1: Tools ---
# ==========================================================================
 
@tool
def load_happiness_data() -> dict:
    """Load and normalize the multi-year World Happiness dataset into memory.
 
    Auto-detects each file's delimiter, standardizes the inconsistent yearly
    column names to canonical lowercase ones, adds a 'year' column from the
    filename, and merges everything into one DataFrame.
 
    Returns:
        dict: 'shape', 'columns', and 'years' on success (plus 'warnings' if any
              file was skipped or had no parseable year), or 'error' on failure.
              NOTE: this returns a dict describing the data, NOT a DataFrame.
    """
    global df
 
    if df is not None:  # cheap idempotency - don't re-read if already loaded
        years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
        return {"shape": list(df.shape), "columns": list(df.columns),
                "years": years, "note": "already loaded (cached)"}
 
    sources, label = _candidate_sources()
    if not sources:
        return {"error": (f"No CSVs in '{LOCAL_DIR}' and no REMOTE_URLS set. "
                          f"Put the yearly files (e.g. 2018.csv) in '{LOCAL_DIR}'.")}
 
    frames, problems = [], []
    for year, src in sources:
        try:
            frame = _read_csv_smart(src)
            frame = _normalize_columns(frame)
            if year is not None:
                frame["year"] = year
            else:
                problems.append(f"{os.path.basename(str(src))}: no 4-digit year in filename "
                                f"- rows from this file will have no 'year'.")
            frames.append(frame)
        except Exception as e:
            problems.append(f"{src}: {type(e).__name__}: {e}")
 
    if not frames:
        return {"error": f"Failed to read any file from {label}. " + " | ".join(problems)}
 
    df = pd.concat(frames, ignore_index=True)
 
    for col in NUMERIC_COLS:                       # coerce numerics; bad values -> NaN
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # Backfill region for any rows missing it, from a country -> region lookup.
    # (Usually unnecessary here since every year carries 'Regional indicator'.)
    if "region" not in df.columns:
        df["region"] = pd.NA
    if "country" in df.columns and df["region"].notna().any():
        region_map = (df.dropna(subset=["region"]).drop_duplicates("country")
                        .set_index("country")["region"])
        df["region"] = df["region"].fillna(df["country"].map(region_map))
 
    years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
    result = {"shape": list(df.shape), "columns": list(df.columns),
              "years": years, "source": label}
    if "year" not in df.columns:
        result["warnings"] = (problems or []) + [
            "No 'year' column - year-based queries (top N by year) will not work. "
            "Name files with a year (e.g. 2018.csv)."]
    elif problems:
        result["warnings"] = problems
    return result
 
 
@tool
def inspect_data(n: int = 5) -> dict:
    """Show dtypes and the first N rows of the loaded dataset.
 
    Use this to SEE the actual data instead of guessing. (load_happiness_data
    returns a dict, not a DataFrame, so you cannot call .head() on its result.)
 
    Args:
        n: number of sample rows to return (default 5).
 
    Returns:
        dict: 'shape', 'dtypes', and 'sample_rows' (list of records) - or 'error'.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    return {"shape": list(df.shape),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample_rows": df.head(int(n)).to_dict(orient="records")}
 
 
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single numeric column.
 
    Args:
        column: The exact (canonical, lowercase) column name to summarize.
 
    Returns:
        dict: count, mean, std, min, quartiles, max - or an 'error'.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": f"Column '{column}' has no numeric values to summarize."}
    return series.describe().to_dict()
 
 
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation and p-value between two numeric columns.
 
    Args:
        col1: Name of the first (canonical, lowercase) column.
        col2: Name of the second (canonical, lowercase) column.
 
    Returns:
        dict: column names, 'pearson_r', 'p_value' (4 dp), 'n' - or an 'error'.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    for c in (col1, col2):
        if c not in df.columns:
            return {"error": f"Column '{c}' not found. Available: {list(df.columns)}"}
    valid = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 2:
        return {"error": "Need at least 2 valid paired data points to correlate."}
    r, p = pearsonr(valid[col1], valid[col2])
    return {"col1": col1, "col2": col2, "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4), "n": int(len(valid))}
 
 
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a column for a specific year.
 
    Args:
        column: The numeric (canonical, lowercase) column to rank by.
        year: The year (integer) to filter on.
        n: How many countries to return (default 5).
 
    Returns:
        dict: 'top_countries' list of {country, value} - or an 'error'.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    for needed in ("year", "country"):
        if needed not in df.columns:
            return {"error": f"Dataset must contain a '{needed}' column."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    year_df = df[df["year"] == year].copy()
    if year_df.empty:
        years = sorted(int(y) for y in df["year"].dropna().unique())
        return {"error": f"No data for year {year}. Years available: {years}"}
    year_df[column] = pd.to_numeric(year_df[column], errors="coerce")
    year_df = year_df.dropna(subset=[column])
    top_n = year_df.sort_values(by=column, ascending=False).head(n)
    return {"top_countries": top_n[["country", column]].to_dict(orient="records")}
 
 
@tool
def aggregate_by_category(group_col: str, value_col: str, agg: str = "mean",
                          year: int = None) -> dict:
    """Aggregate a numeric column grouped by a categorical column.
 
    Answers e.g. "average happiness_score per region in 2020".
 
    Args:
        group_col: Categorical column to group by (e.g. 'region').
        value_col: Numeric column to aggregate (e.g. 'happiness_score').
        agg: One of 'mean', 'median', 'sum', 'min', 'max', 'count'.
        year: Optional year to filter on first; omit to use all years.
 
    Returns:
        dict: 'results' list of {group, value} - or an 'error'.
    """
    global df
    if df is None:
        return {"error": "Data not loaded. Run load_happiness_data first."}
    allowed = {"mean", "median", "sum", "min", "max", "count"}
    if agg not in allowed:
        return {"error": f"agg must be one of {sorted(allowed)}."}
    for c in (group_col, value_col):
        if c not in df.columns:
            return {"error": f"Column '{c}' not found. Available: {list(df.columns)}"}
    data = df
    if year is not None:
        if "year" not in df.columns:
            return {"error": "No 'year' column to filter on."}
        data = df[df["year"] == year]
        if data.empty:
            return {"error": f"No data for year {year}."}
    data = data.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    grouped = (data.dropna(subset=[group_col]).groupby(group_col)[value_col]
                   .agg(agg).round(4).sort_values(ascending=False))
    results = [{"group": k, "value": (None if pd.isna(v) else float(v))}
               for k, v in grouped.items()]
    return {"group_col": group_col, "value_col": value_col, "agg": agg, "results": results}
 
 
# ==========================================================================
# --- Task 2: Build the Agent ---
# ==========================================================================
 
model = OpenAIServerModel(model_id="gpt-4o-mini")
 
SYSTEM_PROMPT = """You are a data analyst assistant for the World Happiness dataset.
 
The data is exposed ONLY through the provided tools. Key facts:
- load_happiness_data() returns a DICT with 'shape', 'columns', 'years'. It is NOT
  a DataFrame: do not call .head(), .columns, or index it like a frame.
- To look at actual rows or dtypes, call inspect_data().
- The canonical, lowercase column names are exactly:
    country, region, happiness_score, gdp_per_capita, life_expectancy, year
  Always use these names (e.g. 'happiness_score', never 'Happiness score').
 
Use summarize_column, compute_correlation, get_top_n_countries, and
aggregate_by_category for analysis. Write Python only when no tool fits (e.g.
custom plots); when you plot, save with plt.savefig(path) then plt.close().
 
CRITICAL: Never invent or recall data from memory. If a tool returns an 'error',
report that error plainly and stop - do NOT fabricate country names, scores, or
statistics. A truthful "the tool failed because X" is always better than a guess.
Be concise and student-friendly."""
 
agent = CodeAgent(
    tools=[load_happiness_data, inspect_data, summarize_column,
           compute_correlation, get_top_n_countries, aggregate_by_category],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "numpy", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)
 
os.makedirs("outputs", exist_ok=True)
 
 
if __name__ == "__main__":
    # Pre-load once and print the result so you can confirm the delimiter and the
    # detected years BEFORE the agent runs (this is where bad data shows up).
    print("--- Pre-loading data ---")
    status = load_happiness_data()
    print(status)
    if "error" in status:
        print("\nData failed to load - fix the source above before running queries.")
    elif not status.get("years"):
        print("\nWARNING: no years detected. Year-based queries will fail. "
              "Check that your filenames contain a 4-digit year.")
 
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]
 
    # reset=True keeps each query's trace clean; the loaded `df` global persists.
    for query in queries:
        print(f"\n--- Query: {query} ---")
        print("AGENT RESPONSE:", agent.run(query, reset=True))
 
    my_query_1 = "Generate a scatter plot of gdp_per_capita vs happiness_score for the year 2019. Save it to outputs/scatter_2019.png."
    print(f"\n--- Custom Query 1: {my_query_1} ---")
    print("AGENT RESPONSE:", agent.run(my_query_1, reset=True))  # forces code generation
 
    my_query_2 = "What were the top 3 countries with the highest life_expectancy in 2018?"
    print(f"\n--- Custom Query 2: {my_query_2} ---")
    print("AGENT RESPONSE:", agent.run(my_query_2, reset=True))  # forces tool use
 
 
# ==========================================================================
# --- Task 5: Reflection ---
# ==========================================================================
"""
--- Reflection ---

1. In Query 3, how did the agent communicate whether the correlation was statistically significant? 
   Did it use the p-value correctly? What threshold did it apply?
   -> The agent correctly looked at the `p_value` returned by the tool. It communicated that the correlation 
      WAS statistically significant by noting that the p-value was essentially 0.0 (or deeply below the standard 
      alpha threshold of 0.05). It understood that a tiny p-value means the relationship is highly unlikely to 
      be due to random chance.

2. Did any of the agent's responses surprise you — either by being more capable than you expected, or less? 
   Describe one specific example.
   -> I was surprised by how seamlessly the CodeAgent handled Query 5 (the multi-line plot). Without being 
      told exactly how to group the data, it knew to use pandas `.groupby('region')` or `.pivot_table()`, 
      isolate the years, and plot it iteratively using matplotlib. It successfully bridged the gap between 
      natural language intent and complex data engineering logic entirely on its own.

3. What one additional tool would make this agent meaningfully more useful? 
   Describe what it would do and what kind of question it would help the agent answer.
   -> An `aggregate_by_category` tool would be massively helpful. It would take a categorical column 
      (like 'region'), a numeric column (like 'happiness_score'), and an aggregator (like 'mean'). 
      This would allow the agent to instantly answer questions like "What is the average happiness score 
      for each region in 2020?" without having to write a raw pandas Python script to execute a `.groupby()` 
      every time.
"""
