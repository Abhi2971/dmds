import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv('student_data.csv')  # Modify based on file path

# Define variables (for example: hours studied vs scores)
x = data[['Hours']].values  # Independent variable
y = data['Scores'].values    # Dependent variable

# Model
model = LinearRegression()
model.fit(x, y)

# Predictions
y_pred = model.predict(x)

# Error metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')
