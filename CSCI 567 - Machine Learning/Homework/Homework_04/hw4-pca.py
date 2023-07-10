import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import math

np.random.seed(0)

#############################
# Read .csv and .txt files #
#############################
def read_csv(path = 'co_occur.csv'):
  """
  Reads a .csv file.

  Input:
      - path: Path of the .csv file

  Return:
      - data: A 10,000 x 10,000 numpy array (the covariance matrix)
  """
  
  data = []
  
  with open(path, 'r') as file:
      
    csvreader = csv.reader(file)
    
    for row in csvreader:
      a = []
      
      for el in row:
        a.append(float(el))
        
      a = np.array(a)
      data.append(a)
      
    data = np.array(data)
    print(np.shape(data))
    
  return data


def read_txt(path):
  """
  Reads a .txt file.

  Input:
      - path: Path of the .txt file

  Return:
      - words: A list of strings (i^th item is the i^th row from the file)
  """
    
  with open(path) as f:
     
    data = f.read()
    words = data.split("\n")
  
    #print(len(words))
 
  return words


##################################
# Plot Eigenvalues (For Part 1) #
##################################
def plot_evs(D):
  '''
  Input: 
     - D: A sequence of eigenvalues
  
  Plots the eigenvalues against their indices and save
  the figure in 'hw4/ev_plot.png'
  '''
  n = np.shape(D)[0]
  x = range(1, n+1)
  fig, ax = plt.subplots()
  ax.plot(x, D)

  ax.set(xlabel = 'Index', ylabel = 'Eigenvalue')
  fig.savefig("W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/ev_plot.png")
  plt.show()


#################################################################################
# Find the Embedding of a Given Word (e.g. 'man', 'woman') (For Parts 2, 4, 5) #
#################################################################################
def find_embedding(word, words, E):
  '''
  Inputs: 
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - emb: The word embedding
  '''  
  ##################################################################
  # TODO: Implement the following steps:
  # i) Find the index/position of 'word' in 'words'.
  for i in range(len(words)):
    if(word == words[i]):
        index = i
        break
  # ii) The row of 'E' at this index will be the embedding 'emb'. 
  emb = E[index]
  ##################################################################
  
  return emb, index

############################################################
# Find the Word Most Similar to a Given Word (For Part 2) #
############################################################
def find_most_sim_word(word, words, E): 
  '''
  Inputs:
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - most_sim_word: The word which is most similar to the input 'word'
  '''
  ###############################################################################################################
  # TODO: Implement the following steps: 
  # i) Get the word embedding of 'word' using the 'find_embedding' function. 
  word_embedding, index = find_embedding(word, words, E)

  # ii) Compute the similarity (dot product) of this embedding with every embedding, i.e. with each row of 'E'.
  similarity_metric = np.random.normal(size=(10000, 1))
  for row in range(len(E)):
        similarity_metric[row] = np.inner(word_embedding,E[row]) 

  # iii) Find the index of the row of 'E' which gives the largest similarity, exculding the row which is the
  # embedding of 'word' from the search (for this, you will need to find the index/position of 'word' in 'words').
  max_similarity = 0
  for row in range(len(similarity_metric)):
    if(row != index):
        if(max_similarity < similarity_metric[row]):
            max_similarity = similarity_metric[row]
            max_index = row

  # iv) Find the word corresponding to that index from 'words'.
  most_sim_word = words[max_index] 
  ###############################################################################################################
    
  return most_sim_word


#############################################################
# Interpret Principal Components/Eigenvectors (For Part 3) #
#############################################################
def find_info_ev(v, words, k=20): 
  # Find words corresponding to the 'k' largest magnitude elements of an eigenvector,
  # to see what kind of information is captured by 'v'.
  '''
  Inputs: 
      - v: An eigenvector (with 10,000 entries)
      - words: The list of all words read from dictionary.txt
      - k: The number of words the function should return
  Return:
      - info: The set of k words 
  '''
  #########################################################################################################################
  # TODO: Implement the following steps:
  # i) Obtain a set of indices (postions in 'v') which would sort the entries of 'v' in decreasing order of absolute value.
  temp = abs(v)
  sorted_indices = sorted(range(len(temp)), key=temp.__getitem__, reverse=True)
  # ii) Using the set of first 'k' indices, get the corresponding words from the list 'words'.
  info = []
  for i in range(20):
    info.append(words[sorted_indices[i]])
  ##########################################################################################################################

  return info 


#################################################################################
# Explore Semantic/Syntactic Concepts Captured by Word Embeddings (For Part 4) #
#################################################################################

def plot_projections(word_seq, proj_seq, filename = 'hw4/projections.png'):
  '''
  Inputs: 
   - word_seq: A sequence of words (strings), used as labels.
   - proj_seq: A sequence of projection values.
   - filename: The plot will be saved 
  
  Plot the set of values in 'proj_seq' on a line with the corresponding labels from 'word_seq and
  save the figure
  '''
  y = np.zeros(len(proj_seq))
  fig, ax = plt.subplots()
  ax.scatter(proj_seq, y)
  
  # to avoid overlaps in the plot
  plt.rcParams.update({'font.size': 9})
  idx = np.argsort(proj_seq)
  proj_seq = proj_seq[idx]
  word_seq = word_seq[idx]
  del_y_prev = 0
  
  for i, label in enumerate(word_seq):
    # to avoid overlaps in the plot
    if i<len(proj_seq)-1:
      if np.abs(proj_seq[i]-proj_seq[i+1])<0.02:
          del_y0 = -0.005-0.0021*len(label)
          if del_y_prev == 0:
            del_y = del_y0
          else:
            del_y = 0
          del_y_prev = del_y0  
      elif (del_y_prev!=0 and del_y == 0 and np.abs(proj_seq[i]-proj_seq[i-1])<0.02):
          del_y = -0.005-0.0021*len(label)
          del_y_prev = del_y
      else:
          del_y = 0
          del_y_prev = 0
          
    ax.text(x = proj_seq[i]-0.01, y = y[i]+0.005+del_y, s = label, rotation = 90)
  
  ax.set_xlim(-0.5,0.5)

  fig.savefig(filename)
  plt.show()  


def get_projections(word_seq, words, E, w):
  '''
  Inputs: 
      - word_seq: A sequence of words (strings)
      - word: The query word (string)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
      - w: An embedding vector (with 10,000 elements)
  Return:
      - proj_seq: The sequence of projections of the words in 'word_seq'
  '''
  ###############################################################################################################
  # TODO: Implement the following steps: 
  # i) For each word in 'word_seq':
  #     - Get its word embedding using the 'find_embedding' function. 
  #     - Find the projection of this embedding onto 'w', i.e. the dot product with normalized 'w'.
  proj_seq = np.random.normal(size = (len(word_seq),1))
  w_e = np.random.normal(size = (len(word_seq), 100)) 
  for i in range (len(word_seq)):
    w_e[i], indexx = find_embedding(word_seq[i], words, E)
  # ii) Return the set of projections. 
  proj_seq = w_e.dot(w.transpose())
  ###############################################################################################################  
  
  return proj_seq
    

#####################################################################
# BONUS: Finding Answers to Analogy Questions (For Part 5 (Bonus)) #
#####################################################################
def find_most_sim_word_w(w, words, E, ids):
  # Find the word whose embedding is most similar to a given vector in the embedding space.
  # Similar to the 'find_most_sim_word' function.
  '''
  Inputs: 
      - w: The query vector (with 100 entries) [for an analogy question, this is a combination of three
                            word embeddings (say words wd1, wd2, wd3), as mentioned in the problem statement]
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
      - ids: The indices of words (wd1, wd2, wd3) which need to be excluded from the search while finding the most similar word
  Return:
      - most_sim_word: The word whose embedding which is most similar to the input 'w'
  '''  
  ########################################################################################################################################
  # TODO: Implement the following steps: 
  # i) Compute the similarity (dot product) of 'w' with every embedding, i.e. with each row of 'E'. 
  similarity_metric = np.random.normal(size=(10000, 1))
  for row in range(len(E)):
        similarity_metric[row] = w.dot(E[row])
  # ii) Find the index of the row of 'E' which gives the largest similarity, exculding the rows at indices given by 'ids' from the search.
  max_similarity = 0
  for row in range(len(similarity_metric)):
    if(row != ids[0] and row != ids[1] and row != ids[2]):
        if(max_similarity < similarity_metric[row]):
            max_similarity = similarity_metric[row]
            max_index = row 
  # iii) Find the word corresponding to that index from 'words'.
  most_sim_word = words[max_index] 
  #########################################################################################################################################
    
  return most_sim_word

  
def find_analog(wd1, wd2, wd3, words, E):
  '''
  Inputs: 
      - wd1, wd2, wd3: Three words (strings) posing an analogy question (e.g. for the one in the problem statement, 
                                                                      these are 'man', 'woman', 'king')
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Return:
      - wd4: The word whose embedding which is most similar to the input 'w'
  '''
  ########################################################################################################################################
  # TODO: Implement the following steps: 
  # i) Find the word embeddings of wd1, wd2 and wd3 using the 'find_embedding' function: w1, w2, w3, respectively.
  aw1, index_wd1 = find_embedding(wd1, words, E)
  aw2, index_wd2 = find_embedding(wd2, words, E)
  aw3, index_wd3 = find_embedding(wd3, words, E)
  # ii) Get vector w = w2 - w1 + w3, and normalize it. 
  aw = aw2 - aw1 + aw3
  aw = np.expand_dims(aw, axis = 0)
  norm_aw = preprocessing.normalize(aw, norm='l1', axis= 1)
  # iii) Find the indices/positions of wd1, wd2, wd3 from the list 'words'.
  # Done in i)
  ids = [index_wd1, index_wd2, index_wd3]
  # iv) Find wd4, the closest word to w (excluding the given three words from the search by using the indices from the previous step), 
  # using the 'find_most_sim_word_w' function. This will be the answer to the analogy question.
  wd4 = find_most_sim_word_w(norm_aw, words, E, ids)
  #########################################################################################################################################
  
  return wd4


def check_analogy_task_acc(task_words, words, E):
  # Check the accuracy for the analogy task by finding answers to a set of analogy questions #
  '''
  Inputs: 
      - task_words: The list of word sequences (wd1, wd2, wd3, wd4), for an analogy statement (e.g. for 
                     the one in the problem statement, these will be 'man', 'woman', 'king', queen)
      - words: The list of all words read from dictionary.txt
      - E: The embedding matrix (10,000 x 100)
  Returns:
      - acc: The average accuracy for the task
  '''
  
  acc = 0
  
  for word_seq in task_words:
    word_seq = word_seq.split()
    wd4_ans = find_analog(word_seq[0], word_seq[1], word_seq[2], words, E)
    if wd4_ans == word_seq[3]:
      acc+=1
  acc = acc/len(task_words)
      
  return acc



################################
# Putting Everything Together #
################################

###########################################################################
# Part 0: Read the files (Make sure the filenames and paths are correct) #
###########################################################################
# M: The co-occurrance matrix
# words: The set of all words in 'dictionary.txt'
# task_words: The set of analogy queries from 'analogy_task.txt' (for the BONUS part)
M = read_csv("W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/co_occur.csv")
words = read_txt("W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/dictionary.txt")
words = np.array(words)
 


###########################################################
# Part 1: Applying PCA to get the low rank approximation #
###########################################################
# TODO:
# i) Apply the log transformation mentioned in the problem statement.
def norm_matrix(data): 
    for row in range(len(data[0])):
        for column in range(len(data[1])):
            data[row][column] = math.log(1 + data[row][column]) 

    return data

# ii) Get centered M, denoted as Mc. First obtain the (d-dimensional) mean feature vector by averaging across all datapoints (rows).
def mean_centered(data):
    meanPoint = data.mean(axis = 1)
    #print(meanPoint)
    for row in range(len(data[0])):
        for column in range(len(data[1])):
            data[row][column] = data[row][column] - meanPoint[column] # Then subtract it from all the n feature vectors.

    return data

norm_M = norm_matrix(M)  # i)
M_c = mean_centered(norm_M)  #ii)



# iii) Use the PCA function (fit method) from the sklearn library to apply PCA on Mc and get its rank-100 approximation (Go through 
# the documentation available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
pca = PCA(n_components=100)
pca.fit(M_c)


# iv) Get the set of eigenvalues and obtain the plot using the 'plot_evs' function provided above.
V = pca.components_  #eigenvectors
eigenvalues = pca.explained_variance_  #eigenvalues
plot_evs(eigenvalues)


# Discussion part to calculated the variance percent
variance = pca.explained_variance_ratio_.cumsum()
print("Variance :{}".format(variance[99]))
#output : 0.7566978267579538



##########################################################
# Part 2: Finding the word most similar to a given word #
##########################################################
# TODO:
# i) Get the set of eigenvectors, i.e. the matrix V (size 10,000 x 100). This will be used in Parts 2, 3.
# Note: Make sure the dimensions are correct (apply transpose, if required).
V_t = np.transpose(V)

# ii) Get P = McV. Normalize the columns of P (to have unit l2-norm) to get E. This will be used in Parts 2, 4, 5.
P = M_c.dot(V_t)
E = preprocessing.normalize(P, norm='l2', axis=0)

# iii) Normalize the rows of E to have unit l2-norm.
norm_E = preprocessing.normalize(P, norm='l2', axis=1)

# iv) Complete the 'find_embedding' function and the 'find_most_sim_word' function.
# Done above!
 
# v) Pick some word(s), e.g. 'university', uqertyse the 'find_most_sim_word' function to find the most similar word.
sim_word = find_most_sim_word('mother', words, norm_E)
print("Most similar word to mother is :{}".format(sim_word))



#########################################################################################
# Part 3: Looking at the information captured by the principal components/eigenvectors #
#########################################################################################
# TODO:
# i) Complete the 'find_info_ev' function, which finsd the words corresponding to k largest magnitude
# elements of a given eigenvector.
# Done above!
# ii) Choose a set of eigenvectors (i.e. columns) from V.
# We will choose the first 5 eigenvectors as it captures the most variance because their eigenvalues values are high.
# iii) For each eigenvector, use the 'find_info_ev' function to see what kind of information it captures. Print the results.
for i in range(5):
  print(find_info_ev(V[i], words))



##############################################################################
# Part 4: Exploring Semantic/Syntactic Concepts Captured by Word Embeddings #
##############################################################################
# TODO:
# i) Use the 'find_embedding' function to find w1 and w2 (the embeddings for 'man' and 'woman', respectively). 
w1, w1_index = find_embedding("woman", words, norm_E)
w2, w2_index = find_embedding("man", words, norm_E)

# ii) Get w = w1 - w2 and normalize it.
w = w1 - w2
w = np.expand_dims(w, axis = 0)
norm_w = preprocessing.normalize(w, norm='l1', axis= 1)

# iii) Complete the 'get_projections' function.
# Done above!

# iv) Use this function to get the set of projections for the list of words mentioned in Part 4.1. 
# Use the 'plot_projections' function to obtain the plot of projections with their corresponding word labels.
# Part 4.1
word_seq = np.array(['boy', 'girl', 'brother', 'sister', 'king', 'queen', 'he', 'she','john', 'mary', 'wall', 'tree'])
proj = get_projections(word_seq, words, norm_E, norm_w)
plot_projections(word_seq, proj, filename = 'W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/projections.png')

# v) Repeat the previous step for the list of words mentioned in Part 4.2 to generate a second plot. You can use the 
# 'filename' argument to make sure the 'plot_projections' function does not overwrite the existing file to save the new plot.
# Part 4.2
word_seq1 = np.array(['math', 'history', 'nurse', 'doctor', 'pilot', 'teacher', 'engineer', 'science', 'arts', 'literature', 'bob', 'alice'])
proj = get_projections(word_seq1, words, norm_E, norm_w)
plot_projections(word_seq1, proj, filename = 'W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/projections1.png')



###########################################################
# Part 5 (BONUS): Finding answers to analogy questions #
###########################################################
# TODO:
# i) Complete the 'find_analog' function.
# Done above!
# ii) Test it using a set of words for an analogy question, such as the one given in the problem statement.
print(find_analog('london', 'england', 'madrid', words, norm_E))
# iii) Use the 'check_analogy_task_acc' function to evaluate 'find_analog' on the list of analogy queries in 'task_words'.
task_words = read_txt('W:/USC/CSCI-567(Machine Learning)/HW4/hw4/hw4/pca/analogy_task.txt')
accuracy = check_analogy_task_acc(task_words, words, norm_E)
print("Accuracy : {}".format(accuracy))

