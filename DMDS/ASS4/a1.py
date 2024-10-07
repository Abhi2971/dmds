# Install the required libraries
# !pip install pandas mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# Assign column names as the dataset doesn't have headers
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Convert to one-hot encoding
df_onehot = pd.get_dummies(df)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_onehot, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the rules along with support and confidence
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
