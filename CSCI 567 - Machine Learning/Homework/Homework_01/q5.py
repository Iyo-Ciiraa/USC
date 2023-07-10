import numpy as np
import matplotlib.pyplot as plt
from cProfile import label
import random

d = 100 # dimensions of data
n = 1000 # number of data points
Xtrain = np.random.normal(0,1, size=(n,d)) 
w_true = np.random.normal(0,1, size=(d,1))
ytrain = Xtrain.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
Xtest = np.random.normal(0,1, size=(n,d))
ytest = Xtest.dot(w_true) + np.random.normal(0,0.5,size=(n,1))

def obj_fn(weight,x,y):
    return ((weight.transpose().dot(np.expand_dims(x, axis=1)))-y)**2 

#==================Problem Set 5.1=======================

Wtrain = np.linalg.inv(Xtrain.transpose().dot(Xtrain)).dot(Xtrain.transpose().dot(ytrain))
Wtest = np.linalg.inv(Xtest.transpose().dot(Xtest)).dot(Xtest.transpose().dot(ytest))

ftrain= 0
ftest= 0
for i in range(n):
   predictortrain = Wtrain.transpose().dot(np.expand_dims(Xtrain[i], axis=1))
   predictortest = Wtest.transpose().dot(np.expand_dims(Xtest[i], axis=1))

   ftrain = ftrain + (predictortrain - ytrain[i])**2 
   ftest = ftest + (predictortest - ytest[i])**2


print("Training RSS : {}".format(ftrain[0]))
print("Testing RSS : {}".format(ftest[0]))
print("The gap: {}".format(ftrain[0]-ftest[0]))


"""
What is the gap in the training and test objective function values? Comment on the result.
->  The gap in the training and test objective function values is known as the generalization gap. This gaps tells us how good a algorithm is at generalizing.
    If the gap is small that means it is better at generalizing which ultimately means that the algo will perform better on unknown data.
    In our solution the gap is very less that means it is good in generalizing.
"""

#==================Problem Set 5.2=======================

N = 20
eta = [0.00005, 0.0005, 0.0007]
testeta = 0.00005
w = [np.random.normal(0,0, size=(d,1))]
J1 = []
J2 = []
J3 = []
for j in range(N):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    sum_grad = np.empty((d,1))
    for i in range(n):
        func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[i], axis=1))-ytrain[i])*np.expand_dims(Xtrain[i], axis=1)
        sum_grad = sum_grad + func_grad
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[0]*sum_grad[k]    
        
    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J1.append(sum)


for j in range(N):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    sum_grad = np.empty((d,1))
    for i in range(n):
        func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[i], axis=1))-ytrain[i])*np.expand_dims(Xtrain[i], axis=1)
        sum_grad = sum_grad + func_grad
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[1]*sum_grad[k]    
    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J2.append(sum)


for j in range(N):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    sum_grad = np.empty((d,1))
    for i in range(n):
        func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[i], axis=1))-ytrain[i])*np.expand_dims(Xtrain[i], axis=1)
        sum_grad = sum_grad + func_grad
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[2]*sum_grad[k]    
    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J3.append(sum)
    

print("Eta = 0.00005 : {}".format(J1))
print("Eta = 0.0005 : {}".format(J2))
print("Eta = 0.0007 : {}".format(J3))  

#Plotting
a = np.linspace(0, 19, 20)   
plt.plot(np.squeeze(J1,axis=1),a.transpose(), label = 'eta = 0.00005')
plt.plot(np.squeeze(J2,axis=1),a.transpose(), label = 'eta = 0.0005')
plt.plot(np.squeeze(J3,axis=1),a.transpose(), label = 'eta = 0.0007')
plt.xlabel('Iterations')
plt.ylabel('Objective Functions')
plt.title('GD Comparison')
plt.legend()
plt.show()
    

"""
Comment in 3-4 sentences on how the step size can affect the convergence of gradient descent.
->  The step size is a hyperparameter which plays crucial part in the convergence of GD. If the stepsize value is very small 
    then it will take more iterations to reach convergence and if it is too big then it might oscillate and finally converge
    or in worst case scenario it may diverge.

Also report the step size that had the best final objective function value and the corresponding objective function value.
->  eta = 0.00005 is the best stepsize because as you see in the graph the cost is decreasing with more number of iterations that means it is converging.
    for the other values the cost is constant which means the gradient is oscillating and not reaching convergence.

"""

#==================Problem Set 5.3=======================
Iter = 1000
eta = [0.00005, 0.005, 0.01]
w = [np.random.normal(0,0, size=(d,1))]
J4 = []
J5 = []
J6 = []
for j in range(Iter):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    ran = random.randrange(1, 1000, 1)
    func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[ran], axis=1))-ytrain[ran])*np.expand_dims(Xtrain[ran], axis=1)
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[0]*func_grad[k]    
        
    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J4.append(sum)


for j in range(Iter):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    ran = random.randrange(1, 1000, 1)
    func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[ran], axis=1))-ytrain[ran])*np.expand_dims(Xtrain[ran], axis=1)
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[0]*func_grad[k]

    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J5.append(sum)


for j in range(Iter):
    last_theta = w[-1]
    this_theta = np.empty((d,1))
    func_grad = np.empty((d,1))
    ran = random.randrange(1, 1000, 1)
    func_grad = (last_theta.transpose().dot(np.expand_dims(Xtrain[ran], axis=1))-ytrain[ran])*np.expand_dims(Xtrain[ran], axis=1)
    for k in range(d):
        this_theta[k] = last_theta[k] - 2*eta[0]*func_grad[k] 

    w.append(this_theta)
    for x in range(n):
        sum = 0
        cost = obj_fn(this_theta,Xtrain[x],ytrain[x])
        sum = sum + cost
    J6.append(sum)

print("Eta = 0.00005 : {}".format(J4))
print("Eta = 0.005 : {}".format(J5))
print("Eta = 0.01 : {}".format(J6))    

    

  

#Plotting
a = np.linspace(0, 999, 1000)   
plt.plot(a.transpose(), np.squeeze(J4,axis=1), label = 'eta = 0.00005')
plt.plot(a.transpose(), np.squeeze(J5,axis=1),label = 'eta = 0.005')
plt.plot(a.transpose(), np.squeeze(J6,axis=1),label = 'eta = 0.01')
plt.xlabel('Iterations')
plt.ylabel('Objective Functions')
plt.title('SGD Comparison')
plt.legend()
plt.show()


"""
Comment 3-4 sentences on how the step size can affect the convergence of stochastic gradient descent and how it compares to gradient descent.
->  The step size is a hyperparameter which plays crucial part in the convergence of SGD. In our case step size with large value is giving us the result.
    For SGD the time taken to converge is inversely propotional to step size. In SGD the convergence is noisy as you can see in the graph but the path of
    convergence does not matter only the final destination to minima matters.
Compare the performance of the two methods.
->  Learning rate of 0.00005 gives the best result for GD.
    Learning rate of 0.00005 gives the best result for SGD.
    For this data distribution GD performs well as the amount of data is less.
How do the best final objective function values compare?
->  The best final objective function values has very large difference beacuse SGD performs better when the data is large and it would help to reduce the computational
    calculations.
How many times does each algorithm use each data point?
->  For GD : Number of Iterations*Number of data points (for our case : 20*1000)
    For SGD : Number of Iterations (i.e 1000)
Also report the step size that had the best final objective function value and the corresponding objective function value.
->  For GD : step size = 0.00005 | objective function value = 0.02863405
    For SGD : step size = 0.00005 | objective function value = 3.48929453
"""