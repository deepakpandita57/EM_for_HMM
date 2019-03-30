#!/usr/bin/python3

# Author: Deepak Pandita
# Date created: 20 Apr 2018

import numpy as np
from scipy.stats import multivariate_normal
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#compute forward probabilities
def forwardProb(A, data, lambda_k, mu, cov):
	N = len(data[0])
	K = len(lambda_k)
	alpha = np.zeros((K,N))
	fsum = np.zeros(N)	#scaling factor

	for i in range(K):
		alpha[i,0] = lambda_k[i] * multivariate_normal.pdf(data[:,0], mu[:,i], cov[i], allow_singular=True)
	fsum[0] = np.sum(alpha[:,0])
	alpha[:,0] = alpha[:,0] / fsum[0]

	for n in range(1,N):
		for i in range(K):
			for j in range(K):
				alpha[i, n] += alpha[j, n-1] * multivariate_normal.pdf(data[:,n], mu[:,i], cov[i]) * A[j, i]
		fsum[n] = np.sum(alpha[:,n])
		alpha[:,n] = alpha[:,n] / fsum[n]
	return alpha, fsum

#compute backward probabilities
def backwardProb(A, data, lambda_k, mu, cov, fsum):
	N = len(data[0])
	K = len(lambda_k)
	beta = np.zeros((K,N))
	beta[:,N-1] = 1

	for n in range(N-2,-1,-1):
		for i in range(K):
			for j in range(K):
				beta[i,n] += beta[j,n+1] * multivariate_normal.pdf(data[:,n+1], mu[:,j], cov[j]) * A[i,j]
		beta[:,n] /= fsum[n+1]

	return beta
	
def main():
	#using optional parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--K', action="store", help = "Value of K", type = int)
	parser.add_argument('--max_iter', action="store", help = "Maximum no. of iterations", type = int)
	args = parser.parse_args()

	#file paths
	data_file = 'points.dat'

	#default no. of clusters and iterations
	K = 2
	max_iter = 15

	if args.K:
		K = args.K
	if args.max_iter:
		max_iter = args.max_iter

	print("K:", K)
	
	#Read data file
	print('Reading data file: '+data_file)
	f = open(data_file)
	lines = f.readlines()
	f.close()

	list = []
	for line in lines:
		temp_list = []
		temp_list.append(float(line.split()[0]))
		temp_list.append(float(line.split()[1]))
		list.append(temp_list)
	
	data = np.array(list)

	#Split the dataset into train and dev
	split_size = int(len(list)*0.9)
	train, dev = data[:split_size].T, data[split_size:].T

	print("Dimensions of Train: ",train.shape)
	
	#transition probability matrix
	A = np.ones((K,K))
	A = A / np.sum(1).T

	#Initialize mean and covariance
	mu = np.random.rand(len(train),K)
	cov = [np.eye(len(train)) for i in range(K)]
	lambda_k = np.ones((K, 1))/K
	
	train_ll=[]
	dev_ll=[]
	old_ll_train = -99999
	old_ll_dev = -99999

	#Start EM algorithm
	for iter in range(max_iter):
		if iter%5==0:
			print('Iteration',iter)
		
		#Compute forward and backward probabilities
		alpha, fsum = forwardProb(A, train, lambda_k, mu, cov)
		beta = backwardProb(A, train, lambda_k, mu, cov, fsum)
		
		#E-Step
		mat = np.zeros((K,K,len(train[0])))
		for n in range(1,len(train[0])):
			mat[:,:,n] = (1/fsum[n])*np.dot(alpha[:,n-1].T, beta[:,n])*A
			for k in range(K):
				mat[:,k,n] *= multivariate_normal.pdf(train[:,n], mu[:,k], cov[k])
		
		#M-Step
		gamma = alpha*beta
		K = np.size(gamma,0)
		#print(K, np.size(train,0))
		lambda_k = gamma[:,0]/np.sum(gamma[:,0]).T
		sum_mat = np.sum(mat[:,:,1:], axis=2)
		A = sum_mat/np.sum(sum_mat, axis=1).T
		
		mu = np.zeros((np.size(train,0),K))
		sum_gamma = np.sum(gamma,axis=1).T
		cov = []
		for k in range(K):  
			mu[:,k] = np.sum(gamma[k,:]*train,axis=1)/sum_gamma[k]
			#print(len(mu), len(mu[0]))
			#print(len(train))
			mu_x = train - mu[:,k][None].T
			cov.append(mu_x.dot((mu_x*(gamma[k,:])).T)/sum_gamma[k])
		
		#Compute log-likelihood
		new_ll_train=np.sum(np.log(fsum))
		dev_alpha, dev_fsum = forwardProb(A, dev, lambda_k, mu, cov)
		new_ll_dev = np.sum(np.log(dev_fsum))
		
		train_ll.append(new_ll_train)
		dev_ll.append(new_ll_dev)
		
		#If the increase in likelihood is less than 1e-6 then stop
		if (new_ll_dev - old_ll_dev) < 1e-6 and new_ll_dev > -np.inf:
			break
		
		old_ll_train = new_ll_train
		old_ll_dev = new_ll_dev
	print('Log-Likelihoods on train:',train_ll)
	print('Log-Likelihoods on dev:',dev_ll)
	
	#Plotting log-likelihood
	plt.figure(1)
	plt.plot(train_ll, label='Train')
	plt.xlabel('Iterations')
	plt.ylabel('LogLikelihood')
	plt.title('Log-Likelihood on Train')
	plt.legend()
	
	plt.figure(2)
	plt.plot(dev_ll, label='Dev')
	plt.xlabel('Iterations')
	plt.ylabel('LogLikelihood')
	plt.title('Log-Likelihood on Dev')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()