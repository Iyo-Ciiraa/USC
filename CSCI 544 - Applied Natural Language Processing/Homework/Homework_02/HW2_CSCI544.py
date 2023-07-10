# %%
import pandas 
import numpy

import nltk

import gensim.downloader
from gensim.models import Word2Vec
from gensim import models
from gensim import utils

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer

# %%
#Setting up cuda environment to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True

#print(torch.cuda.is_available(), torch.__version__)

# %%
#importing data as dataframe
#data = pandas.read_csv('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz', sep='\t', on_bad_lines='skip')
data = pandas.read_csv('data.tsv', sep='\t', on_bad_lines='skip')

# %%
#Remove null value rows and reset index
data = data.dropna()
data = data.reset_index(drop=True)

#Keep only review_body column and corresponding star_rating column
data = data[['review_body', 'star_rating']]

#Removing all non-integer star_rating
data['star_rating'] = data['star_rating'].astype(int)

# %%
#Sample 100000 having 20000 from each star_rating class
data_1 = data[data['star_rating'] == 1].sample(n = 20000, random_state = 1)
data_2 = data[data['star_rating'] == 2].sample(n = 20000, random_state = 1)
data_3 = data[data['star_rating'] == 3].sample(n = 20000, random_state = 1)
data_4 = data[data['star_rating'] == 4].sample(n = 20000, random_state = 1)
data_5 = data[data['star_rating'] == 5].sample(n = 20000, random_state = 1)
dataset = pandas.concat([data_1, data_2, data_3, data_4, data_5])

#print(len(dataset))

# %%
#Loading pre-trained model
model_pre_trained = gensim.downloader.load('word2vec-google-news-300')

# %%
#Check the similarity between two similar words
print('Similarity between excellent and outstanding = ', model_pre_trained.similarity('excellent', 'outstanding'))

#Find the corresponding word given that king - man + woman = queen 
print(model_pre_trained.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))

# %%
#Training my own Word2Vec Model
class MyCorpus:
    def __iter__(self):
        for line in dataset['review_body']:
            yield utils.simple_preprocess(line)

# %%
#Loading my own Word2Vec Model
model_own_trained = models.Word2Vec(sentences = MyCorpus(), vector_size = 300, window = 11, min_count = 10)

# %%
#Check the similarity between two similar words
print('Similarity between excellent and outstanding = ', model_own_trained.wv.similarity('excellent', 'outstanding'))

#Find the corresponding word given that king - man + woman = queen 
print(model_own_trained.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))

# %% [markdown]
# The pretrained model was trained by a much larger corpus (in terms of both quantities and varieties) than my own training dataset. Therefore, the pretrained model encodes better similar words between words than my own model. But since it has such a vast corpus, my model encodes better similarity between words.

# %%
#Function to find the mean of Word2Vec vectors for each review as the input feature

def word2vec_mean(sentences, model):
    sentence_split = sentences.split(' ')
    sum = numpy.zeros(shape = (300,))
    count = 0

    for word in sentence_split:
        if word in model:
            word_vector = model[word]
            sum += word_vector
            count += 1
    mean_vector = sum/count

    return mean_vector

# %%
#Function to concatenate the first 10 Word2Vec vectors for each review as the input feature

def word2vec_concatenation(sentence, model):

  ls = []
  if type(sentence) == list:
    ls = sentence
  if type(sentence) == str:
    ls = sentence.split(' ')

  i = 0 
  j = 0 

  while (i < len(ls)) & (j < 10):
    if ls[i] in model:
      wv = model[ls[i]]
      if j == 0:
        a = wv
      else:
        a = numpy.concatenate((a, wv))
      i += 1
      j += 1
    else:
      i += 1
      
  if j < 10:
    n = 10 - j
    zeros = numpy.zeros(shape=(300*n, ))
    if n == 10:
      a = zeros
    else:
      a = numpy.concatenate((a, zeros))
  return a

# %%
#Function to truncate longer reviews to the length of 20 and padding smaller reviews

def word2vec_sequence(sentence, model):

    if type(sentence) == str:
        ls = sentence.split(' ')
    if type(sentence) == list:
        ls = sentence

    word_vector = []
    for i in range(20):
      try:
        wv = model[ls[i]]
        word_vector.append(wv)
      except:
        pass

    if len(word_vector) < 20:
      for _ in range(20-len(word_vector)):
        word_vector.append([0 for _ in range(300)])
        
    return word_vector

# %%
#Function to get the indexes of NaN in the dataset

def idx_nan(matrix):
  if numpy.any(numpy.isnan(matrix)):
    arr_nan = numpy.argwhere(numpy.isnan(matrix))
    num_nan = arr_nan.shape[0]
    arr = numpy.arange(0, num_nan, 300)
    idx = []
    for i in arr:
      idx.append(arr_nan[i][0])
    return idx
  else:
    return None

# %%
#Splitting data into train and test dataset

labels = dataset['star_rating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(dataset['review_body'], labels, test_size=0.2, random_state=1)

# %%
#Getting average of Word2Vec vectors as input for training and testing data and removing the NaN values

X_train_average = X_train.apply(lambda x: word2vec_mean(x, model_pre_trained))
X_train_pre_trained = numpy.array(X_train_average.values.tolist())

X_test_average = X_test.apply(lambda x: word2vec_mean(x, model_pre_trained))
X_test_pre_trained = numpy.array(X_test_average.values.tolist())

idx_nan_train = idx_nan(X_train_pre_trained)
if idx_nan_train == None:
  y_train_pre_trained = y_train
else:
  X_train_pre_trained = numpy.delete(X_train_pre_trained, idx_nan_train, 0)
  y_train_pre_trained = numpy.delete(y_train, idx_nan_train)

idx_nan_test = idx_nan(X_test_pre_trained)
if idx_nan_test == None:
  y_test_pre_trained = y_test
else:
  X_test_pre_trained = numpy.delete(X_test_pre_trained, idx_nan_test, 0)
  y_test_pre_trained = numpy.delete(y_test, idx_nan_test)

# %%
#Perceptron Analysis

perceptron_pre_trained = Perceptron()
perceptron_pre_trained.fit(X_train_pre_trained, y_train_pre_trained)

prediction_perceptron_pre_trained = perceptron_pre_trained.predict(X_test_pre_trained)

#print(metrics.classification_report(y_test_pre_trained, prediction_perceptron_pre_trained))
print('Perceptron Accuracy = ', accuracy_score(y_test_pre_trained, prediction_perceptron_pre_trained))
print('Perceptron Accuracy using TFIDF Feature Extraction = 0.40675')


# %%
#SVM Analysis

svm_pre_trained = LinearSVC()
svm_pre_trained.fit(X_train_pre_trained, y_train_pre_trained)

prediction_svm_pre_trained = svm_pre_trained.predict(X_test_pre_trained)

#print(metrics.classification_report(y_test_pre_trained, prediction_svm_pre_trained))
print('SVM Accuracy = ', accuracy_score(y_test_pre_trained, prediction_svm_pre_trained))
print('SVM Accuracy using TFIDF Feature Extraction = 0.4897')

# %% [markdown]
# TF-IDF model's performance is better than the Word2vec model because the number of data in each rating class is less.

# %%
#Getting average of Word2Vec vectors as input for training and testing data and removing NaN Values

X_train_nn = X_train.apply(lambda x: word2vec_mean(x, model_pre_trained))
X_train_fnn = numpy.array(X_train_nn.values.tolist())

X_test_nn = X_test.apply(lambda x: word2vec_mean(x, model_pre_trained))
X_test_fnn = numpy.array(X_test_nn.values.tolist())

idx_nan_train = idx_nan(X_train_fnn)
if idx_nan_train != None:
  X_train_fnn_pre_trained = numpy.delete(X_train_fnn, idx_nan_train, 0)
  y_train_fnn_pre_trained = numpy.delete(y_train, idx_nan_train)

idx_nan_test = idx_nan(X_test_fnn)
if idx_nan_test != None:
  X_test_fnn_pre_trained = numpy.delete(X_test_fnn, idx_nan_test, 0)
  y_test_fnn_pre_trained = numpy.delete(y_test, idx_nan_test)

# %%
#Defining the dataset classes

class Train(Dataset):
  def __init__(self, Xtrain, ytrain):
    'Initialization'
    self.data = Xtrain
    self.labels = ytrain

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y

class Test(Dataset):
  def __init__(self, Xtest, ytest):
    'Initialization'
    self.data = Xtest
    self.labels = ytest

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y

# %%
#Creating the training and testing dataset for the FNN Model

train_fnn_pre_trained = Train(X_train_fnn_pre_trained, y_train_fnn_pre_trained - 1)
test_fnn_pre_trained = Test(X_test_fnn_pre_trained, y_test_fnn_pre_trained - 1)

# %%
#Batching and Loading data for the FNN Model

num_workers = 0
batch_size = 100
valid_size = 0.2

num_train = len(train_fnn_pre_trained)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader_fnn = torch.utils.data.DataLoader(train_fnn_pre_trained, batch_size=batch_size,
                                           sampler=train_sampler, num_workers
                                           =num_workers)
valid_loader_fnn = torch.utils.data.DataLoader(train_fnn_pre_trained, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers
                                           =num_workers)
test_loader_fnn = torch.utils.data.DataLoader(test_fnn_pre_trained, batch_size=batch_size,
                                           num_workers=num_workers)

# %%
#Defining the MLP Architecture

class ThreeLayerMLP(torch.nn.Module):
  def __init__(self, D_in, H1, H2, D_out):
    super().__init__()
    self.linear1 = torch.nn.Linear(D_in, H1)
    self.linear2 = torch.nn.Linear(H1, H2)
    self.linear3 = torch.nn.Linear(H2, D_out)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    h1_relu = torch.nn.functional.relu(self.linear1(x))
    h1_drop = self.dropout(h1_relu)
    h2_relu = torch.nn.functional.relu(self.linear2(h1_drop))
    h2_drop = self.dropout(h2_relu)
    h2_output = self.linear3(h2_drop)

    return h2_output

# %%
#Initializing the FNN Model

model_fnn = ThreeLayerMLP(300, 50, 10, 5)
model_fnn.cuda()
print(model_fnn)

# %%
#Loading the parameters for the FNN Model

criterion = torch.nn.CrossEntropyLoss()
optimizer_fnn = torch.optim.SGD(model_fnn.parameters(), lr=0.0065)

# %%
#Training the FNN Model

n_epochs = 200

valid_loss_min = numpy.Inf 

for epoch in range(n_epochs):

  train_loss = 0.0
  valid_loss = 0.0

  model_fnn.train() 
  for data, target in train_loader_fnn:
    target = target.type(torch.LongTensor) 
    data, target = data.to(device), target.to(device)
    optimizer_fnn.zero_grad()
    output = model_fnn(data.float())
    loss = criterion(output, target)
    loss.backward()
    optimizer_fnn.step()
    train_loss += loss.item()*data.size(0)

  model_fnn.eval() 
  for data, target in valid_loader_fnn:
    target = target.type(torch.LongTensor) 
    data, target = data.to(device), target.to(device)
    output = model_fnn(data.float())
    loss = criterion(output, target)
    valid_loss += loss.item()*data.size(0)

  train_loss = train_loss/len(train_loader_fnn.dataset)
  valid_loss = valid_loss/len(valid_loader_fnn.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))
  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
        .format(valid_loss_min, valid_loss))
    torch.save(model_fnn.state_dict(), 'model_fnn.pt')
    valid_loss_min = valid_loss

# %%
#Load the model with the lowest validation loss
model_fnn.load_state_dict(torch.load('model_fnn.pt'))

# %%
#Evaluating the FNN Model

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader_fnn:
    embeddings, labels = data
    model_fnn.to("cpu")
    outputs = model_fnn(embeddings.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))

# %%
#Using the function word2vec_concatenation to get values for the datasets and removing NaN values

X_train_fnn_10_val = X_train.apply(lambda x: word2vec_concatenation(x, model_pre_trained))
X_train_fnn_10 = numpy.array(X_train_fnn_10_val.values.tolist())

X_test_fnn_10_val = X_test.apply(lambda x: word2vec_concatenation(x, model_pre_trained))
X_test_fnn_10 = numpy.array(X_test_fnn_10_val.values.tolist())

idx_nan_train = idx_nan(X_train_fnn_10)
if idx_nan_train != None:
    X_train_fnn_10_pre_trained = numpy.delete(X_train_fnn_10, idx_nan_train, 0)
    y_train_fnn_10_pre_trained = numpy.delete(y_train, idx_nan_train)
else:
    X_train_fnn_10_pre_trained = X_train_fnn_10
    y_train_fnn_10_pre_trained = y_train

idx_nan_test = idx_nan(X_test_fnn_10)
if idx_nan_test != None:
    X_test_fnn_10_pre_trained = numpy.delete(X_test_fnn_10, idx_nan_test, 0)
    y_test_fnn_10_pre_trained = numpy.delete(y_test, idx_nan_test)
else:
    X_test_fnn_10_pre_trained = X_test_fnn_10
    y_test_fnn_10_pre_trained = y_test

# %%
#Defining the dataset classes

class Train(Dataset):
  def __init__(self, xtrain, ytrain):
    'Initialization'
    self.data = xtrain
    self.labels = ytrain

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y
class Test(Dataset):
  def __init__(self, xtest, ytest):
    'Initialization'
    self.data = xtest
    self.labels = ytest

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y

# %%
#Creating the training and testing dataset for FNN Model

train_fnn_10 = Train(X_train_fnn_10_pre_trained, y_train_fnn_10_pre_trained-1)
test_fnn_10 = Test(X_test_fnn_10_pre_trained, y_test_fnn_10_pre_trained-1)

# %%
#Batching and Loadind data for the FNN Model

num_workers = 0
batch_size = 100
valid_size = 0.2

num_train = len(train_fnn_10)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader_fnn_10 = torch.utils.data.DataLoader(train_fnn_10, batch_size=batch_size,
                                           sampler=train_sampler, num_workers
                                           =num_workers)
valid_loader_fnn_10 = torch.utils.data.DataLoader(train_fnn_10, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers
                                           =num_workers)
test_loader_fnn_10 = torch.utils.data.DataLoader(test_fnn_10, batch_size=batch_size,
                                           num_workers=num_workers)

# %%
#Defining the MLP Architecture and loading the model

class ThreeLayerMLP(nn.Module):
  def __init__(self, D_in, H1, H2, D_out):
    super().__init__()
    self.linear1 = nn.Linear(D_in, H1)
    self.linear2 = nn.Linear(H1, H2)
    self.linear3 = nn.Linear(H2, D_out)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    h1_relu = F.relu(self.linear1(x))
    h1_drop = self.dropout(h1_relu)
    h2_relu = F.relu(self.linear2(h1_drop))
    h2_drop = self.dropout(h2_relu)
    h2_output = self.linear3(h2_drop)

    return h2_output
    
model_fnn_10 = ThreeLayerMLP(3000, 50, 10, 5)
model_fnn_10.to(device)
print(model_fnn_10)

# %%
#Specifying the parameters

criterion = nn.CrossEntropyLoss()
optimizer_fnn_10 = torch.optim.SGD(model_fnn_10.parameters(), lr=0.007)

# %%
#Training the FNN Model

n_epochs = 100

valid_loss_min = numpy.Inf 

for epoch in range(n_epochs):
  train_loss = 0.0
  valid_loss = 0.0

  model_fnn_10.train() 
  for data, target in train_loader_fnn_10:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    optimizer_fnn_10.zero_grad()
    output = model_fnn_10(data.float())
    loss = criterion(output, target)
    loss.backward()
    optimizer_fnn_10.step()
    train_loss += loss.item()*data.size(0)

  for data, target in valid_loader_fnn_10:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    output = model_fnn_10(data.float())
    loss = criterion(output, target)
    valid_loss += loss.item()*data.size(0)

  train_loss = train_loss/len(train_loader_fnn_10.dataset)
  valid_loss = valid_loss/len(valid_loader_fnn_10.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))

  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
        .format(valid_loss_min, valid_loss))
    torch.save(model_fnn_10.state_dict(), 'model_fnn_10.pt')
    valid_loss_min = valid_loss

# %%
#Load the model with the lowest validation loss
model_fnn_10.load_state_dict(torch.load('model_fnn_10.pt'))

# %%
#Evaluating the FNN Model

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader_fnn_10:
    embeddings, labels = data
    model_fnn_10.to("cpu")
    outputs = model_fnn_10(embeddings.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy: %d %%' % (100 * correct / total))

# %% [markdown]
# The accuracies of FNN is better as compared to the simple models. This is so because of the learning algorithms for FNN. 

# %%
#Calculating values for training and testing dataset using word2vec_sequence and removing NaN values

X_train_rnn_val = X_train.apply(lambda x: word2vec_sequence(x, model_pre_trained))
X_train_rnn = numpy.array(X_train_rnn_val.values.tolist())

X_test_rnn_val = X_test.apply(lambda x: word2vec_sequence(x, model_pre_trained))
X_test_rnn = numpy.array(X_test_rnn_val.values.tolist())

idx_nan_train = idx_nan(X_train_rnn)
if idx_nan_train != None:
  X_train_rnn_pre_trained = numpy.delete(X_train_rnn, idx_nan_train, 0)
  y_train_rnn_pre_trained = numpy.delete(y_train, idx_nan_train)
else:
  X_train_rnn_pre_trained = X_train_rnn
  y_train_rnn_pre_trained = y_train

idx_nan_test = idx_nan(X_test_rnn)
if idx_nan_test != None:
  X_test_rnn_pre_trained = numpy.delete(X_test_rnn, idx_nan_test, 0)
  y_test_rnn_pre_trained = numpy.delete(y_test, idx_nan_test)
else:
  X_test_rnn_pre_trained = X_test_rnn
  y_test_rnn_pre_trained = y_test

# %%
#Creating the training and testing dataset

train_rnn_pre_trained = Train(X_train_rnn_pre_trained, y_train_rnn_pre_trained - 1)
test_rnn_pre_trained = Test(X_test_rnn_pre_trained, y_test_rnn_pre_trained - 1)

# %%
#Batching and Loading data for the RNN Model

num_workers = 0
batch_size = 100
valid_size = 0.2

num_train = len(train_rnn_pre_trained)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader_rnn = torch.utils.data.DataLoader(train_rnn_pre_trained, batch_size=batch_size,
                                           sampler=train_sampler, num_workers
                                           =num_workers)
valid_loader_rnn = torch.utils.data.DataLoader(train_rnn_pre_trained, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers
                                           =num_workers)
test_loader_rnn = torch.utils.data.DataLoader(test_rnn_pre_trained, batch_size=batch_size,
                                           num_workers=num_workers)

# %%
#Defining the RNN Architecture

class RNNModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                     nonlinearity='relu')
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    
    h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
    out, hn = self.rnn(x, h0)
    out = self.fc(out[:, -1, :])
    return out

# %%
#Initializing the RNN Model

model_rnn = RNNModel(300, 20, 1, 5)
model_rnn.cuda()
print(model_rnn)

# %%
#Loading the Parameters for the RNN Model

optimizer_rnn = torch.optim.SGD(model_rnn.parameters(), lr=0.0075)

# %%
#Training the RNN Model

n_epochs = 200

valid_loss_min = numpy.Inf 

for epoch in range(n_epochs):

  train_loss = 0.0
  valid_loss = 0.0

  model_rnn.train() 

  for data, target in train_loader_rnn:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    optimizer_rnn.zero_grad()
    output = model_rnn(data.float())
    loss = criterion(output, target)
    loss.backward()
    optimizer_rnn.step()
    train_loss += loss.item()*data.size(0)

  model_rnn.eval() 

  for data, target in valid_loader_rnn:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    output = model_rnn(data.float())
    loss = criterion(output, target)
    valid_loss += loss.item()*data.size(0)
    

  train_loss = train_loss/len(train_loader_rnn.dataset)
  valid_loss = valid_loss/len(valid_loader_rnn.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))

  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
        .format(valid_loss_min, valid_loss))
    torch.save(model_rnn.state_dict(), 'model_rnn.pt')
    valid_loss_min = valid_loss

# %%
#Load the model with the lowest validation loss
model_rnn.load_state_dict(torch.load('model_rnn.pt'))

# %%
#Evaluating the RNN Model

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader_rnn:
    embeddings, labels = data
    embeddings, labels = embeddings.to(device), labels.to(device)
    model_rnn.to(device)
    outputs = model_rnn(embeddings.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
print('Accuracy of RNN Model: %d %%' % (100 * correct / total))

# %% [markdown]
# RNN works better when the dataset provided is large, but since we are working on smaller batches and data, the accuracy of FNN is much better as compared to RNN.

# %%
#Using the function word2vec_sequence to get values for training and testing data, and removing any NaN values

X_train_gru_val = X_train.apply(lambda x: word2vec_sequence(x, model_pre_trained))
X_train_gru = numpy.array(X_train_gru_val.values.tolist())

X_test_gru_val = X_test.apply(lambda x: word2vec_sequence(x, model_pre_trained))
X_test_gru = numpy.array(X_test_gru_val.values.tolist())

idx_nan_train = idx_nan(X_train_gru)
if idx_nan_train != None:
  X_train_gru_pre_trained = numpy.delete(X_train_gru, idx_nan_train, 0)
  y_train_gru_pre_trained = numpy.delete(y_train, idx_nan_train)
else:
  X_train_gru_pre_trained = X_train_gru
  y_train_gru_pre_trained = y_train

idx_nan_test = idx_nan(X_test_gru)
if idx_nan_test != None:
  X_test_gru_pre_trained = numpy.delete(X_test_gru, idx_nan_test, 0)
  y_test_gru_pre_trained = numpy.delete(y_test, idx_nan_test)
else:
  X_test_gru_pre_trained = X_test_gru
  y_test_gru_pre_trained = y_test

# %%
#Defining the data classes

class Train(Dataset):
  def __init__(self, xtrain, ytrain):
    'Initialization'
    self.data = xtrain
    self.labels = ytrain

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y
class Test(Dataset):
  def __init__(self, xtest, ytest):
    'Initialization'
    self.data = xtest
    self.labels = ytest

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.data)

  def __getitem__(self, index):
    'Generates one sample of data'
    X = self.data[index]
    y = self.labels[index]

    return X, y

# %%
#Creating the training and testing dataset

train_gru_pre_trained = Train(X_train_gru_pre_trained, y_train_gru_pre_trained - 1) 
test_gru_pre_trained = Test(X_test_gru_pre_trained, y_test_gru_pre_trained - 1)

# %%
#Batching and Loading the Data for the GRU Model

num_workers = 0
batch_size = 100
valid_size = 0.2

num_train = len(train_gru_pre_trained)
indices = list(range(num_train))
numpy.random.shuffle(indices)
split = int(numpy.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader_gru = torch.utils.data.DataLoader(train_gru_pre_trained, batch_size=batch_size,
                                           sampler=train_sampler, num_workers
                                           =num_workers)
valid_loader_gru = torch.utils.data.DataLoader(train_gru_pre_trained, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers
                                           =num_workers)
test_loader_gru = torch.utils.data.DataLoader(test_gru_pre_trained, batch_size=batch_size,
                                           num_workers=num_workers)

# %%
#Defining the GRU Architecture

class GRUModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
    out, hn = self.gru(x, h0)
    out = self.fc(out[:, -1, :])
    return out

# %%
#Initializing the GRU Model

model_gru = GRUModel(300, 20, 1, 5)
model_gru.cuda()
print(model_gru)

# %%
#Setting the GRU Parameters

criterion = nn.CrossEntropyLoss()
optimizer_gru = torch.optim.SGD(model_gru.parameters(), lr=0.0075)

# %%
#Training the GRU Model

n_epochs = 100

valid_loss_min = numpy.Inf 

for epoch in range(n_epochs):

  train_loss = 0.0
  valid_loss = 0.0

  model_gru.train() 

  for data, target in train_loader_gru:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    optimizer_gru.zero_grad()
    output = model_gru(data.float())
    loss = criterion(output, target)
    loss.backward()
    optimizer_gru.step()
    train_loss += loss.item()*data.size(0)

  model_gru.eval() 

  for data, target in valid_loader_gru:
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    output = model_gru(data.float())
    loss = criterion(output, target)
    valid_loss += loss.item()*data.size(0)

  train_loss = train_loss/len(train_loader_gru.dataset)
  valid_loss = valid_loss/len(valid_loader_gru.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, valid_loss))
    
  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
        .format(valid_loss_min, valid_loss))
    torch.save(model_gru.state_dict(), 'model_gru.pt')
    valid_loss_min = valid_loss

# %%
#Load the model with the lowest validation loss

model_gru.load_state_dict(torch.load('model_gru.pt'))

# %%
#Evaluating the GRU Model

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader_gru:
    embeddings, labels = data
    embeddings, labels = embeddings.to(device), labels.to(device)
    model_gru.to(device)
    outputs = model_gru(embeddings.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('GRU Accuracy: %d %%' % (100 * correct / total))

# %% [markdown]
# The flow of information is controlled and the problem of long delays is completely eliminated in the Gated RNN's, thus, the accuracy is lower as compared to RNN.


