# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the data
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', )
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', )
test_set = np.array(test_set, dtype='int')


# Getting the number of users and movies
nb_users = max(max(training_set[:, 0]), max(test_set[:, 0]))
nb_movies = max(max(training_set[:, 1]), max(test_set[:, 1]))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self,):
        """Module의 SAE를 상속받음"""
        super(SAE, self).__init__()
        """autoencoder와 연결돼있는 full connection"""
        """Linear(input vector, hidden neuron 수)"""
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        """더이상 encoding layer를 추가하지 않기때문에 제일 처음의 input vector수와 같게 hidden neuron수를 구성한다.
           = decoding"""
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    """auto encodiner 안에서의 action을 정의함"""
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
epochs = 200
for epoch in range(1, epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            """기울기 계산"""
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            """평가 된 영화의 평점의 평균을 구함"""
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
            """backward와 step의 차이
               backward : 방향
               step : 질량"""
    print('epoch:', epoch, 'loss:', (train_loss/s))
    
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    """input을 training_set으로 놔둔 이유 : training set으로 예측된 항목이 test set에 들어있어서 같이 비교할 수 있기때문"""
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        loss.backward()
        test_loss += np.sqrt(loss.data*mean_corrector)
        s+=1.
        optimizer.step()
    print('loss:', (test_loss/s))
    