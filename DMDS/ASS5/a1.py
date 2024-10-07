import numpy as np

# Data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 16, 18])

# Mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the coefficients
b1 = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

print(f'Estimated coefficients: b0 = {b0}, b1 = {b1}')
