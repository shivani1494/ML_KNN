import cPickle, gzip
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import random

class KNN:

	def __init__(sf):
		file = open("train.txt")
		train_data = file.readlines()
		train_data = [x.strip() for x in train_data]
		
		#print train_data
	
		file = open("test.txt")
		test_data = file.readlines()
		test_data = [x.strip() for x in test_data]

	
		#print test_data
						
		file = open("validate.txt")
		validate_data = file.readlines()
		validate_data = [x.strip() for x in validate_data]


		#print validate_data
						


if __name__ == '__main__':
	
	knn = KNN()
