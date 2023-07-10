import numpy as np
import json
import collections
import matplotlib.pyplot as plt

def data_processing(data):
	
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-Max scaling
	
	def minmax(x):
		for i in range(x.shape[0]):
			x[i] = (x[i] - x[i].min()/ (x[i].max() - x[i].min()))
		return x

	if do_minmax_scaling:
		pass
		Xtrain = minmax(Xtrain)
		Xval = minmax(Xval)
		Xtest = minmax(Xtest)

	# Normalization
	def normalization(x):
		
		for i in range(x.shape[0]):
			x[i] = (x[i] / np.sqrt(np.sum(x[i]) ** 2))
		return x
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	dists = np.zeros(shape=(X.shape[0], Xtrain.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Xtrain.shape[0]):
			dists[i, j] = np.sqrt(np.sum((X[i] - Xtrain[j]) ** 2))

	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	dists = np.zeros(shape=(X.shape[0], Xtrain.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Xtrain.shape[0]):
			xi = np.sqrt(np.sum((X[i]) ** 2))
			xj = np.sqrt(np.sum((Xtrain[j]) ** 2))
			if (xi == 0):
				dists[i, j] = 1
			elif (xj == 0):
				dists[i, j] = 1
			else:
				numerator = np.dot(X[i], Xtrain[j])
				denominator = xi * xj
				dists[i, j] = 1 - (numerator/denominator)

	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	ypred = np.zeros(shape=(dists.shape[0],))
	for i in range(dists.shape[0]):
		index = np.argsort(dists[i])[:k]
		labels = np.array([ytrain[ind] for ind in index])
		unique, counts = np.unique(labels, return_counts=True)
		counter = dict(zip(unique, counts))
		ypred[i] = max(counter, key=counter.get)
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	err = (y != ypred).sum() / len(y)
	return err


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	validation_error = []
	for k in K:
		ypred = predict_labels(k, ytrain, dists)
		validation_error.append(compute_error_rate(yval, ypred))

	best_err = np.sort(validation_error)[0]
	i = np.argsort(np.array(validation_error))[0]
	best_k = K[i]
	return best_k, validation_error, best_err


def main():
	input_file = 'disease.json'
	output_file = 'output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()
	
	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	err_training = []
	dists = compute_l2_distances(Xtrain, Xtrain)
	print("Training Set:")
	for k in K:
		ypred = predict_labels(k, Xtrain, dists)
		error_per_k = compute_error_rate(ytrain, ypred)
		err_training.append(error_per_k)
		print("k =", k, ":", error_per_k)
	print()
	err_validation = []
	dists = compute_l2_distances(Xval, Xval)
	print("Validation Set:")
	for k in K:
		ypred = predict_labels(k, Xval, dists)
		error_per_k = compute_error_rate(yval, ypred)
		err_training.append(error_per_k)
		print("k =", k, ":", error_per_k)
	print()
	#plt.plot(err_training, label = 'Training Error Curve')
	#plt.legend()
	#plt.show()

	#plt.plot(err_validation, label = 'Validation Error Curve')
	#plt.legend()
	#plt.show()

	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)
	 
	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print()
	print("Using the best k, the final test error rate is", test_err)
	print()
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

if __name__ == "__main__":
	main()
