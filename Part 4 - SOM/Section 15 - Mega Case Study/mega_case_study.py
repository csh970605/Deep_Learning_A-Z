# Mega Case Study - Make a Hybrid Deep Learning Model

# Part 1 - Identify the Frauds with the Self_organizing Map
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
som = MiniSom(x=10, y=10, input_len=len(X[1]), sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
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
frauds = np.concatenate((mappings[6,5], mappings[(6,7)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from Unsuperviesd to Supervised Deep Learning

# Creating the Matrix of features
customers = dataset.iloc[:, 1:].values


# Creating the Dependent Variable
"""모든 고객이 속이지 않은것으로 한다(0).
   fraud에서 CustomerId를 추출하여 걔네만 1로한다"""
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Part 3 - ANN

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Import Tensorflow
import tensorflow as tf

# Initalizing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(customers, is_fraud, batch_size=1, epochs=10)

#Predicting test set results
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
print(y_pred)