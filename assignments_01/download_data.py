import urllib.request
import os

# Create exact folder
target_dir = "../assignments/resources/happiness_project"
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading files to {target_dir}...")

# Loop and download each CSV
for year in range(2015, 2025):
    # This is the "Raw" URL format for GitHub files
    url = f"https://raw.githubusercontent.com/Code-the-Dream-School/python-200/main/assignments/resources/happiness_project/world_happiness_{year}.csv"
    
    file_path = os.path.join(target_dir, f"world_happiness_{year}.csv")
    
    print(f"Fetching {year} data...")
    urllib.request.urlretrieve(url, file_path)

print("\nSuccess! All 10 files have been downloaded.")