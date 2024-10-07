import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load inbuilt dataset (Iris dataset as an example)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Display shapes of the training and testing sets
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
