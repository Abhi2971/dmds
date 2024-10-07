import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Step 1: Load dataset from GitHub (replace URL with the actual dataset URL)
url = "https://raw.githubusercontent.com/path-to-your-dataset.csv"
data = pd.read_csv(url)

# Step 2: Handling missing values
print("Null values before removing:\n", data.isnull().sum())
data.fillna(data.mean(), inplace=True)  # Fill missing values with column mean

# Step 3: Encode categorical values to numeric
categorical_columns = ['categorical_column']
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 4: Splitting dataset into features (X) and target (y) (assuming the last column is the target)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Step 5: Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature scaling (rescale between 0 and 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display preprocessed data
print(f"Training data (scaled):\n {X_train_scaled[:5]}")
print(f"Test data (scaled):\n {X_test_scaled[:5]}")
