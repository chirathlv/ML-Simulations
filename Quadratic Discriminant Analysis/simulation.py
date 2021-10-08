# Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import  load_iris

# input data
data = load_iris()
x = data.data[:, :2] # Chose only 2 variables
y = data.target.reshape(-1,1) # Target variable
data = np.concatenate((x,y), axis=1) # Concatenate data together
np.random.shuffle(data) # Shuffle the data

for i, _ in enumerate(data):
    if i > 10: # Num of classes has to be more than 2 and should have enough data to cal Cov Metces, so skipping some data points 
        # Model
        model = QuadraticDiscriminantAnalysis()
        model.fit(data[:i+1, :2], data[:i+1, 2])
        score = model.score(data[:i+1, :2], data[:i+1, 2])

        # Mesh Grid
        x_1_min, x_1_max, x_2_min, x_2_max = data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_1_min - 1, x_1_max + 1, 0.1), np.arange(x_2_min - 1, x_2_max + 1, 0.1))

        # Concatenate and predictions for the grid points
        z = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1)
        zz = model.predict(z).reshape(xx.shape)

        # Decision Boundries
        plt.title("Quadratic Discriminant Analysis Classifier Simulation", fontweight='bold')
        plt.xlim(x_1_min, x_1_max)
        plt.ylim(x_2_min, x_2_max)
        plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA']))

        # Score and points labels
        plt.text(7.2, 4.3, "Points : " + str(i+1), fontweight='bold')
        plt.text(7.2, 4.2, "Score : " + str(round(score, 2)), fontweight='bold')

        # Actual Data
        plt.scatter(data[:i+1, 0], data[:i+1, 1], c=data[:i+1, 2], cmap=ListedColormap(['#FF0000', '#0000FF', '#00FF00']))
        plt.pause(0.1)
        plt.clf()

plt.show()
