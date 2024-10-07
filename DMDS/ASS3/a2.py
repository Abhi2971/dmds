import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Create the CSV manually or load from a file
data = {
    'Age': [35, 42, 25, 32, 38],
    'Experience': [10, 12, 4, 8, 7],
    'Rank': [7, 6, 9, 8, 5],
    'Nationality': ['UK', 'USA', 'N', 'USA', 'UK'],
    'Go': ['NO', 'YES', 'YES', 'NO', 'YES']
}
df = pd.DataFrame(data)

# Convert categorical data to numerical
df['Nationality'] = df['Nationality'].apply(lambda x: 1 if x == 'USA' else 0)
X = df.drop('Go', axis=1)
y = df['Go']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict for a 40-year-old American comedian, with 10 years of experience, and a comedy rank of 7
new_show = [[40, 10, 7, 1]]  # Age, Experience, Rank, Nationality(1 for USA)
prediction = clf.predict(new_show)

print(f'Prediction for new show: {prediction[0]}')

# Visualize the tree (optional)
tree.plot_tree(clf)
