
import numpy as np
import matplotlib.pyplot as plt


heights = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
weights = np.array([53, 55, 59, 61, 65, 68, 70, 74, 76])

n = len(heights)

sum_x = np.sum(heights)
sum_y = np.sum(weights)
sum_x2 = np.sum(heights ** 2)
sum_xy = np.sum(heights * weights)

w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
w0 = (sum_y - w1 * sum_x) / n


print(f'Regression Coefficient w0 (Intercept): {w0:.2f}')
print(f'Regression Coefficient w1 (Slope): {w1:.2f}')


height_to_predict = 5.9
predicted_weight = w0 + w1 * height_to_predict
print(f'Predicted weight for height {height_to_predict} feet: {predicted_weight:.2f} pounds')

predicted_weights = w0 + w1 * heights

errors = weights - predicted_weights

mse = np.mean(errors ** 2)
print(f'Mean Squared Error (MSE): {mse:.2f}')

plt.figure(figsize=(10, 6))


plt.scatter(heights, weights, color='blue', label='Data Points')


regression_line_x = np.linspace(min(heights), max(heights), 100)
regression_line_y = w0 + w1 * regression_line_x
plt.plot(regression_line_x, regression_line_y, color='red', label='Regression Line')


plt.xlabel('Height (feet)')
plt.ylabel('Weight (pounds)')
plt.title('Height vs. Weight Linear Regression')
plt.legend()
plt.grid(True)

plt.show()