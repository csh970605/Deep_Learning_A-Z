# Self Organizing Map

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
"""x, y는 만들어질 map의 가로와 세로를 뜻함
   input_len : 입력할 feature의 수"""
som = MiniSom(x=10, y=10, input_len=len(X[1]), sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
"""num_iteration : 몇번 반복할지"""
som.train_random(data=X, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
"""distance_map : SOM의 winning node의 MID를 구함"""
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(X)
"""plot에서 가장 밝은 부분의 좌표를 concat함"""
frauds = np.concatenate((mappings[(3,1)], mappings[(4,1)]), axis=0)
frauds = sc.inverse_transform(frauds)