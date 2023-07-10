import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
np.random.seed(42)

#Global varibales and datasets
d = 100 # dimensions of data
n = 1000 # number of data points
hf_train_sz = int(0.8 * n//2)
X_pos = np.random.normal(size=(n//2, d))
X_pos = X_pos + .12
X_neg = np.random.normal(size=(n//2, d))
X_neg = X_neg - .12
X_train = np.concatenate([X_pos[:hf_train_sz],X_neg[:hf_train_sz]])
X_test = np.concatenate([X_pos[hf_train_sz:],X_neg[hf_train_sz:]])
y_train = np.concatenate([np.ones(hf_train_sz),-1 * np.ones(hf_train_sz)])
y_test = np.concatenate([np.ones(n//2 - hf_train_sz),-1 * np.ones(n//2 - hf_train_sz)])

#Functions to calculate values
def obj_fn(weight,x,y):
    return m.log(1 + m.exp(-y*weight.transpose().dot(np.expand_dims(x, axis=1))))

def negSig(z):
    return m.exp(-z)/(1 + m.exp(-z))

def obj_fn01(f):
    if (f<=0):
        return 1
    else:
        return 0


#==================Problem Set 4.1=======================

Iter = 5000
eta = [0.0005, 0.005, 0.05]
w = [np.random.normal(0,0, size=(d,1))]
Jtrain = np.empty(shape = (3,5000))
Jtest = np.empty(shape = (3,5000))

#Training data
for e in range(3):
    for j in range(Iter):
        last_theta = w[-1]
        this_theta = np.empty((d,1))
        ran_train = random.randrange(1, 800, 1)
        X_i = np.expand_dims(X_train[ran_train], axis=1)
        sig = negSig(last_theta.transpose().dot(X_i)*y_train[ran_train])
        #print(sig)
        #print(ran_train)
        this_theta = last_theta + (eta[e]*sig*y_train[ran_train])*X_i   
            
        w.append(this_theta)
        sum = 0
        for x in range(len(X_train)):
            cost = obj_fn(this_theta,X_train[x],y_train[x])
            sum = sum + cost

        Jtrain[e][j] = sum

#Test Data
for e in range(3):
    for j in range(Iter):
        last_theta = w[-1]
        this_theta = np.empty((d,1))
        ran_test = random.randrange(1, 200, 1)
        X_i = np.expand_dims(X_test[ran_test], axis=1)
        sig = negSig(last_theta.transpose().dot(X_i)*y_test[ran_test])
        #print(sig)
        #print(ran_train)
        this_theta = last_theta + (eta[e]*sig*y_train[ran_train])*X_i   
            
        w.append(this_theta)
        sum = 0
        for x in range(len(X_test)):
            cost = obj_fn(this_theta,X_test[x],y_test[x])
            sum = sum + cost

        Jtest[e][j] = sum


#Plotting
a = np.linspace(0, 4999, 5000)   
plt.plot(a.transpose(), Jtrain[0], label = 'Train - eta = 0.0005')
plt.plot(a.transpose(), Jtrain[1], label = 'Train - eta = 0.005')
plt.plot(a.transpose(), Jtrain[2], label = 'Train - eta = 0.05')
plt.plot(a.transpose(), Jtest[0], label = 'Test - eta = 0.0005')
plt.plot(a.transpose(), Jtest[1], label = 'Test - eta = 0.005')
plt.plot(a.transpose(), Jtest[2], label = 'Test - eta = 0.05')
plt.xlabel('Iterations')
plt.ylabel('Objective Functions')
plt.title('SGD on Logistic Loss')
plt.legend()
plt.show()



"""
How do the objective function values on the train and test data relate with each other for different step sizes? Comment in 3-4 sentences.
->  As you can see for eta=0.05 on test data the value of objective function keeps on increasing which means the step size is too large and SGD is not converging.
But if you plot individually for train and test data the value of objective function keeps on decreasing for eta=0.0005 & 0.005. The best result we get is for 
eta = 0.005 where the objective function value is least after 5000 iterations on both the train and test data. 
"""

#==================Problem Set 4.2=======================

lowest_value = 1000000
eta_value = 0
Jtest_01 = np.empty(shape = (3,5000))
for e in range(3):
    for j in range(Iter):
        last_theta = w[-1]
        this_theta = np.empty((d,1))
        ran_test = random.randrange(1, 200, 1)
        X_i = np.expand_dims(X_test[ran_test], axis=1)
        identity = obj_fn01(last_theta.transpose().dot(X_i)*y_test[ran_test])
        #print(sig)
        #print(ran_train)
        this_theta = last_theta - (eta[e]*identity*y_train[ran_train])*X_i   
            
        w.append(this_theta)
        #f = this_theta.transpose().dot(X_test[x])*y_test[x]
        sum = 0
        for x in range(len(X_test)):
            cost = obj_fn01(this_theta.transpose().dot(X_test[x])*y_test[x])
            sum = sum + cost
        sum = sum/len(X_test)    

        Jtest_01[e][j] = sum
    
    print("Cost of Objective Function at 5000 iteration for eta = {} is {} ".format(eta[e],sum))

    if(sum<lowest_value):
        lowest_value = sum
        eta_value = eta[e]

print("The lowest value of objective function {} for eta = {}".format(lowest_value,eta_value))
        

#Plotting
a = np.linspace(0, 4999, 5000)   
plt.plot(a.transpose(), Jtest_01[0], label = 'eta = 0.0005')
plt.plot(a.transpose(), Jtest_01[1],label = 'eta = 0.005')
plt.plot(a.transpose(), Jtest_01[2],label = 'eta = 0.05')
plt.xlabel('Iterations')
plt.ylabel('Objective Functions')
plt.title('SGD on 0-1 Loss')
plt.legend()
plt.show()


#==================Problem Set 4.3=======================
"""
Comment on how well the logistic loss act as a surrogate for the 0-1 loss.
->  As you can see the loss value for 0-1 loss is way less then logistic loss but the surrogate loss indicates the goodness of the classifier.  
As minimizing the 0-1 loss is hard we use the surrogate losses to calculate the loss as these losses can minimized (as you can see in the plot).
"""