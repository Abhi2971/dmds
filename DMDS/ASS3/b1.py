from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Dataset
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# Convert categorical data to numerical
weather_map = {'Sunny': 0, 'Overcast': 1, 'Rainy': 2}
temp_map = {'Hot': 0, 'Mild': 1, 'Cool': 2}
play_map = {'No': 0, 'Yes': 1}

weather_num = np.array([weather_map[i] for i in weather])
temp_num = np.array([temp_map[i] for i in temp])
play_num = np.array([play_map[i] for i in play])

# Features and target
X = np.column_stack((weather_num, temp_num))
y = play_num

# Naive Bayes Classifier
clf = CategoricalNB()
clf.fit(X, y)

# Prediction for [Overcast, Mild]
test_instance = [[1, 1]]  # 1: Overcast, 1: Mild
pred = clf.predict(test_instance)

# Convert prediction back to categorical
play_result = 'Yes' if pred[0] == 1 else 'No'
print(f'Prediction for the tuple [Overcast, Mild]: {play_result}')
