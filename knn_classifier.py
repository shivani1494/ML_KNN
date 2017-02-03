import cPickle, gzip
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import random
import Queue as Q
class KNN:

	def __init__(sf):
		file = open("train.txt")
		sf.train_data = file.readlines()
		print "sf.train_data: ", sf.train_data		
		for x in sf.train_data:
			x.strip()
			x = x.split()
			x = np.array(x)
			x.astype(int)
			#print "x: ", x
		
		sf.train_data = [x.strip() for x in sf.train_data]
		
		for x in range(len(sf.train_data)):
			

		sf.train_data = np.array(sf.train_data)
		
		print sf.train_data
		#print train_data
	
		file = open("test.txt")
		sf.test_data = file.readlines()

		for x in sf.test_data:
			x.strip()
			x = x.split()
			x = np.array(x)
			x.astype(int)
			print "x: ", x
		
		sf.test_data = [x for x in sf.test_data]
		sf.test_data = np.array(sf.test_data)
		
	
		#print test_data
						
		file = open("validate.txt")
		sf.validate_data = file.readlines()
		sf.validate_data = [x.strip() for x in sf.validate_data]
		sf.validate_data = np.array(sf.validate_data)
		sf.validate_data.astype(int)		

		#print validate_data
						
	
	def one_knn(sf):
		label = sf.knn_computation(1)
		print label
	def five_knn(sf):
		label = sf.knn_computation(5)
	
	def nine_knn(sf):
		label = sf.knn_computation(9)
	
	def fifteen_knn(sf):
		label = sf.knn_computation(15)
	
	def knn_computation(sf, k, data):		
		
		minDist = -1
		label = -1
		if(k == 0):
			minDist = INT_MAX
		else:
			k_q = Q.PriorityQueue()
			
		
		for td in range(len(data)):
			for t in range(len(sf.train_data)):		
				euclidean_dist = sf.eucledian_distance(sf.train_data[t], data[td])		
				if( not k ):
					if(minDist > euclidean_dist):
						minDist = euclidean_dist
						label = sf.train_data[t][784]
				else:
					if(k_q.size() == k):
						dist_top = k_q.get()[0]
						#make sure this logic is working well
						if(dist_top < (-1*euclidean_dist)):
							k_q.pop()
					k_q.put((-1*euclidean_dist, sf.train_data[t][784]))

					#do some computation here for k nearest neighbors

		return label
	
	def eucledian_distance(sf, x1, x2):
		print "x1", x1
		print "x2", x2
		return np.sqrt(sum((x1-x2)**2));	
			
	def run_knn(sf):
		sf.knn_computation(1, sf.test_data)

		#sf.knn_computation(5)

		#sf.knn_computation(9)

		#sf.knn_computation(15)
	
	#def calc_training_error(sf):

	#def calc_testing_error(sf):
	
	#def calc_validation_error(sf):

if __name__ == '__main__':
	
	knn = KNN()
	knn.run_knn()
