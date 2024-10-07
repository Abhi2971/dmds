import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample car dataset from W3Schools (you can replace this with actual data)
data = pd.DataFrame({
    'Weight': [2300, 2500, 2700, 3000, 3200],
    'Volume': [1300, 1500, 1600, 1700, 1800],
    'CO2': [89, 95, 99, 100, 105]
})

x = data[['Weight', 'Volume']]  # Independent variables
y = data['CO2']  # Dependent variable

# Model
model = LinearRegression()
model.fit(x, y)

# Coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
