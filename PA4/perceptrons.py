import numpy as np
import copy

### DEFAULT FILENAMES
TRAINING_NAME = "hw4train.txt"
TEST_NAME = "hw4test.txt" 
DICTIONARY_NAME = "hw4dictionary.txt"

class Perceptron:

	#make sure everything is in numpy
	def __init__(sf):

		sf.input_data, sf.input_label = sf.read_data(TRAINING_NAME)

		sf.test_data, sf.test_label = sf.read_data(TEST_NAME)

		#shape of input_data == 2000, 891
		#initialize to all 0's
		sf.weight_mat = np.zeros(sf.input_data.shape[1])

		#save weigth count and weights for voted and averaged perceptrons
		sf.weight_count = []
		sf.all_weight_mat = []

		#for averaged perceptron algo
		sf.running_avg = np.zeros(sf.weight_mat.shape)
		
		sf.train_err = []

		sf.test_err = []

	def read_data(sf, filename):
		#read test train and label data

		print("Loading %s ..." %filename)

		filehandle = open(filename, "r")
		line = filehandle.readline()
		data = []
		
		while line != "":
			line = line.split()
			data += [line]
			line = filehandle.readline()
		
		data = np.array(data)
		data = data.astype(float)

		print(filename,"loaded : Dim",data.shape)
		return data[:,:-1], data[:,-1]

	def perceptron(sf, data, label, test_data, test_label):

		num_passes = 1	

		#is the hyperplane getting updated every time
		#we compute a new weight vector?
		#such that the weigth vector is always normla to the 
		#hyperplane?

		#print "data.shape: ", data.shape
		#print "label.shape: ", label.shape

		for t in range(num_passes):         
			for i in range(data.shape[0]):
				
				dot_XW = np.dot(data[i], sf.weight_mat)
				#print "dot prod: ", dot_XW
				#print "sign of dot prod: ", np.sign(dot_XW)
				if(label[i]*dot_XW <= 0):
					sf.weight_mat = sf.weight_mat + ( label[i]*data[i] )
					#print "i label[i]: ", i, ": ", label[i]

				#if (i == 0):
				#print "weights: ", sf.weight_mat
				

			sf.train_err += [sf.run_perceptron(data, label)] 
			print("train err after", t+1 ,"pass:", sf.train_err[-1])
			sf.test_err +=  [sf.run_perceptron(test_data, test_label)]
			print("test err after", t+1, "pass:", sf.test_err[-1])

	'''	
	def extract_labelAB(i, j):

		data = []
		label = []
		for x in range(len(sf.input_label)):
			
			if(input_label[x] == 1 || input_label[x] == 2):
				data += 
	'''


	def run_perceptron(sf, data, label):
		#predict output for normal perceptron

		err = 0.0
		for i in range(data.shape[0]):
			dot_YW = np.dot(data[i], sf.weight_mat)
			class_sign = np.sign(dot_YW)

			if(class_sign != label[i]):
				err += 1

		return err/data.shape[0]


	def voted_perceptron(sf, data, label, test_data, test_label):
		count = 1
		num_passes = 3

		for t in range(num_passes):        
			for i in range(len(data)):
				dot_XW = np.dot(data[i], sf.weight_mat)
				if(label[i]*dot_XW <= 0):
					temp_weight_mat = sf.weight_mat + (label[i]*data[i])
					
					#append for the previous matrix
					sf.all_weight_mat.append(copy.copy(sf.weight_mat))
					sf.weight_count.append(count)

					sf.weight_mat = temp_weight_mat 
					count = 1
					
				else:
					count += 1

			sf.run_voted_perceptron(t)

			sf.train_err += [run_voted_perceptron(data, label)]
			print("train err after", t+1, "pass:", sf.train_err[t])
			sf.test_err += [run_voted_perceptron(test_data, test_label)]
			print("train err after", t+1, "pass:", sf.test_err[t])


	def run_voted_perceptron(sf, data, label):

		sum_sign = 0
		for t in range(len(test_data)):

			#you can totally vectorize this loop
			for i in range(len(sf.all_weight_mat)):

				dot_WY = np.dot(sf.all_weight_mat[i]*test_data[t])
				sum_sign += sf.weight_count[i]*np.sign(dot_WY)

				#final sign or class of test data t
			class_t = np.sign(sum_sign)

			if(class_t != label[t]):
				err += 1

		return err/data.shape[0]

	#think about why would voted and averaged perceptron give you the
	#same result?!
	def averaged_perceptron(sf):
		count = 1
		num_passes = 4
		for t in range(num_passes):         
			for i in range(len(sf.input_data)):
				dot_XW = np.dot(sf.input_data[i], sf.weight_mat)
				if(sf.label[i]*dot_XW <= 0):
					temp_weight_mat = sf.weight_mat + (sf.label[i]*sf.input_data[i])
		
					#append for the previous matrix
					sf.running_avg += sf.weight_mat*count
					sf.weight_mat = temp_weight_mat

					err += 1
					count = 1        
				else:
					count += 1

			sf.run_averaged_perceptron(t)

			sf.train_err += run_averaged_perceptron(sf.input_data, sf.input_label)
			print("train err after", t+1, "pass:", sf.train_err[-1])
			sf.test_errt += run_averaged_perceptron(sf.test_data, sf.test_label)
			print("train err after", t+1, "pass:", sf.test_err[-1])


	def run_averaged_perceptron(sf, data, label):

		sum_sign = 0
		for t in range(len(data)):

			#you can totally vectorize this loop
			dot_WY = np.dot(sf.running_avg*data[t])

			#final sign or class of test data t
			class_t = np.sign(dot_WY)

			if(class_t != label[t]):
				err += 1 

		sf.test_err[t] = sf.calculate_testing_error(err)
		print("train err after", t+1, "pass:", sf.test_err[t])

		return err/data.shape[0]


	def classify_A_VS_B(sf, a, b=None, allLabels=None):

		if(allLabels == None):
			labels_ind_a =  np.where(sf.input_label == a)[0]
			labels_ind_b =  np.where(sf.input_label == b)[0]
		
		if(b == None):
			labels_ind_a =  np.where(sf.input_label == a)[0]
			labels_ind_b =  np.where(sf.input_label != a)[0]
		
		#print "labels_inx_1: ", labels_ind_1
		#print "labels_idx_2: ", labels_ind_2
		
		data = np.vstack((sf.input_data[labels_ind_a,:],sf.input_data[labels_ind_b,:]))

		#print "data[3]: ", data[3]
		label_a = sf.input_label[labels_ind_a].reshape(len(labels_ind_a),1)		
		label_a[:,0] = 1

		label_b = sf.input_label[labels_ind_b].reshape(len(labels_ind_b),1)
		label_b[:,0] = -1

		#print "labels_1[690]: ", label_1
		#print "labels_2[690], ", label_2

		label = np.vstack((label_a,label_b))

		#print "label[3]: ", label[3]

		data_label = np.hstack((data, label))

		np.random.shuffle(data_label)

		print "data_label ", data_label

		data = data_label[:, :-1]
		print "data ", data		
		label = data_label[:, -1]
		print "label ", label

		labels_ind_a =  np.where(sf.test_label == 1)[0]
		labels_ind_b =  np.where(sf.test_label == 2)[0]
		
		data_test = np.vstack((sf.test_data[labels_ind_a,:],sf.test_data[labels_ind_b,:]))

		label_a = sf.test_label[labels_ind_a].reshape(len(labels_ind_a),1)
		label_a[:,0] = 1
		label_b = sf.test_label[labels_ind_b].reshape(len(labels_ind_b),1)
		label_b[:,0] = -1
		label_test = np.vstack((label_a,label_b))

		sf.perceptron(data, label, data_test, label_test)

if __name__ == '__main__':

	ptrn = Perceptron()
	ptrn.classify_1vs2()
	#ptrn.perceptron()

	#X_T, Y_T = ptrn.read_data(TEST_NAME)
	#print(X.shape,Y.shape)    
		
	#ptrn.voted_perceptron()
	#ptrn.averaged_perceptron()

