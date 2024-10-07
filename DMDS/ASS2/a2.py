import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "https://raw.githubusercontent.com/path-to-your-dataset.csv"
data = pd.read_csv(url)

# Select categorical columns (replace 'categorical_column' with your column name)
categorical_columns = ['categorical_column']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert each categorical column to numeric
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Check the result
print(data.head())

# Save the transformed dataset (optional)
data.to_csv('numeric_dataset.csv', index=False)
