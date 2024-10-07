# Install the required libraries
# !pip install pandas mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
url = "https://github.com/amankharwal/Website-data/raw/master/Groceries%20dataset.csv"
df = pd.read_csv(url)

# Convert to one-hot encoding
df_onehot = df.pivot_table(index='Member_number', columns='itemDescription', aggfunc='size', fill_value=0)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_onehot, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the rules along with support and confidence
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
