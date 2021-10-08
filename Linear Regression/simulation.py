# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

# inputs
data_points = int(input("How many data points? "))
dispersion = int(input("Dispersion of the data (1 - 10): "))

# input data
x = np.linspace(1, 10, data_points).reshape(-1, 1)
# Random Gaussian Error
e = np.random.randn(data_points).reshape(-1, 1) * dispersion
# Output data
y = 5 * x + e
# X, Y data
data = np.concatenate((x, y), axis=1)
# Shuffle data
np.random.shuffle(data)

# Liner Regression Model


def model(data, i):
    # reshape the data for the model
    x = data[:i+1, 0].reshape(-1, 1)
    y = data[:i+1, 1].reshape(-1, 1)

    # defining the model
    model = lr()
    model.fit(x, y)
    y_pred = model.predict(data[:, 0].reshape(-1, 1))

    # Minimum of 2 data points required to calculate R²
    score = model.score(x, y) if i > 2 else 0
    return y_pred, score

# Plotting function


def plot(data, y_pred, score, i):
    plt.title('Linear Regression Simulation', fontweight='bold')
    plt.xlim(1, 10)
    plt.ylim(-50, 100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(data[:i+1, 0], data[:i+1, 1], color='blue')
    plt.plot(data[:, 0], y_pred, color='red')
    plt.text(1, 90, ' Points : ' + str(i+1),
             color='Green', fontweight='bold')
    plt.text(1, 82, ' R² : '+str(round(score, 2)),
             color='Green', fontweight='bold')  # R² score
    plt.pause(0.1)
    plt.clf()


# Simulation
for i, _ in enumerate(data):
    # model predictions
    predictions, score = model(data, i)

    # plotting data and predictions
    plot(data, predictions, score, i)

plt.show()
