import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load inbuilt dataset (Iris dataset as an example)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Rescale data between 0 and 1
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Display the scaled data
print(scaled_data.head())
