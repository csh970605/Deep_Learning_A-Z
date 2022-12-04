# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


# Feature Scaling
"""정규화 사용"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
""" number of timesteps : 
    다음 주가를 예측할 때 RNN이 기억해야할 사항을 명시하는 데이터 구조
   n timestep : 마지막 ~ 마지막 - n개의 데이터 """
X_train = []
y_train = []
for i in range(60, len(training_set)): # 60은 timesteps의 수와 같음
    X_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
"""reshape 함수에서 가장 마지막의 1은 넣을 인자가 1개뿐이기 때문에 1임
    n개면 n을 넣으면 됨"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
"""연속된 값을 예측하기 때문에 regression을 이용함"""
regressor = Sequential() 

# Adding the first LSTM layer and some Dropout regularisation
"""units : LSTM이 각 층마다 가질 뉴런의 수
   return_sequence : LSTM층 뒤에 다른 층을 추가할떄 : true / 추가안하면 false
   input _hape : timesteps(X_train.shape[1])"""
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
"""rate : 층마다 탈락시킬 뉴런의 수"""
regressor.add(Dropout(rate=0.2))

# Adding a second LSTM layer and some Dropout regularisation
"""input_shape를 지정하지 않아도 되는 이유 : 첫번째 LSTM층에서 input_shape를 제공하기 때문"""
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
"""return_sequence의 default가 False이므로 지워도 무관"""
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
"""RNN에서는 optimizer로 RMSProp을 keras에서 추천하지만 adam은 항상 안전한 optimizer이기 떄문에 선택함"""
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
"""예측할 날짜의 처음-60을 구하는 작업"""
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
"""dataframe으로 만드는 과정"""
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 60 + len(dataset_test)):
    X_test.append(inputs[i-60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()