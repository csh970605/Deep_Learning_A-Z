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
"""\t는 sep보다는 delimiter를 사용한다"""
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t', )
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t', )
test_set = np.array(test_set, dtype='int')


# Getting the number of users and movies
nb_users = max(max(training_set[:, 0]), max(test_set[:, 0]))
nb_movies = max(max(training_set[:, 1]), max(test_set[:, 1]))

# Converting the data into an array with users in lines and movies in columns
"""Torch에서는 numpy array대신 list를 사용한다"""
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        """id_movies : i번째 user가 평가한 영화의 데이터가 담긴 list"""
        """id_ratings : i번째 user가 평가한 평점의 데이터가 담긴 list"""
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        """평가 하지 않은 영화는 0을 입력한다"""
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
"""FloatTensor : data가 float형이기 떄문에 single type tensor를 생성하기 위한 method"""
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Converting the ratings into binary ratings 1(Liked) or 0 (Not LIked)
"""0과 1로 변환하는 이유 : 평가받지 않은 rating을 계산하기 위한 RBM에게 일관된 data를 제공하기 위해"""
"""데이터셋의 0을 -1로 변환하는 이유 : 평가하지 않은 항목이기 때문에"""
"""평점이 1인것과 2인것을 나눈 이유 : torch tensor는 or를 인식하지 못함"""
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 1] = 0
training_set[training_set >= 3] = 0
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 0

#Creating the architecture of the Neural Network
class RBM():
    """nv : visible node
       nh : hidden node"""
    def __init__(self, nv, nh):
        """weight을 임의로 초기화"""
        self.W = torch.randn(nh, nv)
        """bias(편향) 초기화"""
        """1 : batch를 위한 fake dimension
           nh : 편향
           torch는 2차원 vector를 받기 때문에 fakedimension을 만듦"""
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    """visible node에 의해 hidden node가 활성화 될 확률을 구하는 함수
       x : hidden nueron의 확률"""
    def sample_h(self, x):
        """w는 hidden node의 pv이기 때문에 뒤바꿀 필요 O"""
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    """hidden node에 의해 visible node가 활성화 될 확률을 구하는 함수"""
    def sample_v(self, y):
        """w는 hidden node의 pv이기 때문에 뒤바꿀 필요 X"""
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    """v0 : input vector(user의 평가)
       VK : k sampling이후 생성되는 visible node
       ph : 처음 hidden node의 probability
       phk : k sampling이후의 probability"""
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)         

nv = len(training_set[0])
nh = 100 # Tunable
"""batch_size : batch size만큼의 observation 후 update를 함"""
batch_size = 100 #Tunable
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    """batch size만큼 user를 observation한 후 weight를 업데이트함"""
    for id_user in range(0, nb_users - batch_size, batch_size):
        """vk : 만들어지는 visual node
           v0 : 최초 visual node"""
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        """Gibbs sampling """
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch:', epoch,  'loss:', (train_loss/s))
    
# Testing the RBM
"""Markov chain Monte Carlo(MCMC) technique 사용"""
test_loss = 0
s = 0.
for id_user in range(nb_users):
    """v를 test_set으로 바꾸지 않는 이유는 RBM의 hidden neuron을 활성화 하는역할로는 사용할 수 있기때문"""
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss:', (test_loss/s))