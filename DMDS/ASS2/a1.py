import pandas as pd

# Step 1: Load dataset from GitHub (replace URL with the actual dataset URL)
url = "https://raw.githubusercontent.com/path-to-your-dataset.csv"
data = pd.read_csv(url)

# Step 2: Check for null values
print("Null values before removing:\n", data.isnull().sum())

# Step 3: Remove rows with null values
cleaned_data = data.dropna()

# Step 4: Check for null values after removal
print("\nNull values after removing:\n", cleaned_data.isnull().sum())

# Save the cleaned dataset (optional)
cleaned_data.to_csv('cleaned_dataset.csv', index=False)
