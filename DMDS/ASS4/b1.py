# Install the required libraries
# !pip install pandas

import pandas as pd

# Load the dataset
url = "https://www.kaggle.com/spscientist/students-performance-in-exams?select=StudentsPerformance.csv"
df = pd.read_csv(url)

# a) Display the shape of dataset
print("Shape of dataset:", df.shape)

# b) Display the top rows of the dataset with their columns
print("Top rows of the dataset:\n", df.head())

# c) Display the number of rows randomly
random_rows = df.sample(n=5)
print("Randomly sampled rows:\n", random_rows)

# d) Display the number of columns and names of the columns
print("Number of columns:", df.shape[1])
print("Names of columns:", df.columns)
