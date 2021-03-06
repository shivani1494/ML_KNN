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

		sf.dict = sf.read_data(DICTIONARY_NAME, True)

		#shape of input_data == 2000, 891
		#initialize to all 0's
		
		sf.weight_mat_1 = np.zeros(sf.input_data.shape[1])
		sf.weight_mat_2 = np.zeros(sf.input_data.shape[1])
		sf.weight_mat_3 = np.zeros(sf.input_data.shape[1])
		sf.weight_mat_4 = np.zeros(sf.input_data.shape[1])
		sf.weight_mat_5 = np.zeros(sf.input_data.shape[1])
		sf.weight_mat_6 = np.zeros(sf.input_data.shape[1])
		
		sf.num_classes = 6
		sf.conf_mat = np.zeros((sf.num_classes+1,sf.num_classes))

		sf.train_err = []
		sf.test_err = []

	def read_data(sf, filename, isDict=None):
		#read test train and label data

		print("Loading %s ..." %filename)

		filehandle = open(filename, "r")
		line = filehandle.readline()
		data = []
		
		while line != "":
			line = line.split()
			data += [line]
			line = filehandle.readline()
		
		if(isDict == True):
			return data

		data = np.array(data)
		data = data.astype(float)

		print(filename,"loaded : Dim",data.shape)
		return data[:,:-1], data[:,-1]


	def perceptron(sf, data, label, test_data, test_label, weight_mat):

		num_passes = 2

		print("Running Regular Perceptron!")

		for t in range(num_passes):         
			for i in range(data.shape[0]):
				
				dot_XW = np.dot(data[i], weight_mat)
				#print "dot prod: ", dot_XW
				#print "sign of dot prod: ", np.sign(dot_XW)
				if(label[i]*dot_XW <= 0):
					weight_mat = weight_mat + ( label[i]*data[i] )
					#print "i label[i]: ", i, ": ", label[i]

				#if (i == 0):
				#print "weights: ", weight_mat				

			sf.train_err += [sf.test_perceptron(data, label, weight_mat)] 
			print("train err after", t+1 ,"pass:", sf.train_err[-1])
			sf.test_err +=  [sf.test_perceptron(test_data, test_label, weight_mat)]
			print("test err after", t+1, "pass:", sf.test_err[-1])
		return weight_mat


	def run_one_vs_all(sf):

		#C1
		data1, label1, data_test1, label_test1 = sf.get_data_AvsB(1)
		
		sf.weight_mat_1 = sf.perceptron(data1, label1, data_test1, label_test1, sf.weight_mat_1)

		#C2
		data2, label2, data_test2, label_test2 = sf.get_data_AvsB(2)
		
		sf.weight_mat_2 = sf.perceptron(data2, label2, data_test2, label_test2, sf.weight_mat_2)

		#C3
		data3, label3, data_test3, label_test3 = sf.get_data_AvsB(3)
		
		sf.weight_mat_3 = sf.perceptron(data3, label3, data_test3, label_test3, sf.weight_mat_3)
		
		#C4
		data4, label4, data_test4, label_test4 = sf.get_data_AvsB(4)		
		
		sf.weight_mat_4 = sf.perceptron(data4, label4, data_test4, label_test4, sf.weight_mat_4)

		#C5
		data5, label5, data_test5, label_test5 = sf.get_data_AvsB(5)
		
		sf.weight_mat_5 = sf.perceptron(data5, label5, data_test5, label_test5, sf.weight_mat_5)
		
		#C6
		data6, label6, data_test6, label_test6 = sf.get_data_AvsB(6)
		
		sf.weight_mat_6 = sf.perceptron(data6, label6, data_test6, label_test6, sf.weight_mat_6)

		results = sf.test_one_vs_all(sf.input_data,sf.input_label)
		results_test = sf.test_one_vs_all(sf.test_data,sf.test_label)
		print("training err",results[1])
		print("test err",results_test[1])

		sf.confusion_matrix(results_test[0], sf.test_label)
		print(sf.conf_mat)

	def confusion_matrix(sf, predictions, labels):
		classes = [1,2,3,4,5,6,-1]
		for i in range(len(classes)):
			for j in range(len(classes[:-1])):
				ind_j = np.where(labels == classes[j])[0]
				C = np.count_nonzero(predictions[ind_j] == classes[i])
				N = len(ind_j) * 1.0
				sf.conf_mat[i,j] = C/N

	def get_data_AvsB(sf, a, b=None):
		# a = 1
		# b = -1
		all_data = np.hstack((sf.input_data, sf.input_label.reshape((sf.input_label.shape[0],1))))
		all_test_data = np.hstack((sf.test_data, sf.test_label.reshape((sf.test_label.shape[0],1))))

		if b == None:
			# b is the rest of labels
			for i in range(all_data.shape[0]):
				if all_data[i,-1] == a:
					all_data[i,-1] = 1
				else:
					all_data[i,-1] = -1

			for i in range(all_test_data.shape[0]):
				if all_test_data[i,-1] == a:
					all_test_data[i,-1] = 1
				else:
					all_test_data[i,-1] = -1
		else:
			# b is a label
			rows_not_used = []
			for i in range(all_data.shape[0]):
				if all_data[i,-1] == a:
					all_data[i,-1] = 1
				elif all_data[i,-1] == b:
					all_data[i,-1] = -1
				else:
					rows_not_used += [i]
			all_data = np.delete(all_data,rows_not_used,0)

			rows_not_used = []
			for i in range(all_test_data.shape[0]):
				if all_test_data[i,-1] == a:
					all_test_data[i,-1] = 1
				elif all_test_data[i,-1] == b:
					all_test_data[i,-1] = -1
				else:
					rows_not_used += [i]
			all_test_data = np.delete(all_test_data, rows_not_used,0)

		return (all_data[:,:-1], all_data[:,-1], all_test_data[:,:-1], all_test_data[:,-1])


	def test_one_vs_all(sf, data, labels):
		
		predictions = []
		for i in range(data.shape[0]):
			class_sign = [0, 0, 0, 0, 0, 0]	
			#class 1
			dot_YW = np.dot(data[i], sf.weight_mat_1)

			class_sign[0] = np.sign(dot_YW)

			#class 2
			dot_YW = np.dot(data[i], sf.weight_mat_2)
			class_sign[1] = np.sign(dot_YW)

			#class 3
			dot_YW = np.dot(data[i], sf.weight_mat_3)
			class_sign[2] = np.sign(dot_YW)

			#class 4
			dot_YW = np.dot(data[i], sf.weight_mat_4)
			class_sign[3] = np.sign(dot_YW)

			#class 5
			dot_YW = np.dot(data[i], sf.weight_mat_5)
			class_sign[4] = np.sign(dot_YW)

			#class 6
			dot_YW = np.dot(data[i], sf.weight_mat_6)
			class_sign[5] = np.sign(dot_YW)


			count_class = 0
			class_label = 0
			dont_know = False
			no_class = True

			for c in range(len(class_sign)):
				
				if class_sign[c] == 1:					
					count_class += 1
					class_label = c+1
					no_class = False

				if count_class > 1:
					dont_know = True
					break

			if dont_know:
				label = -1
			elif no_class:
				label = -1
			else:
				label = class_label
			predictions += [label]

		err = (data.shape[0]-np.count_nonzero(np.array(predictions) == labels))/data.shape[0]
		return np.array(predictions), err

	def test_perceptron(sf, data, label, weight_mat):
		#predict output for normal perceptron

		err = 0.0
		for i in range(data.shape[0]):
			dot_YW = np.dot(data[i], weight_mat)
			class_sign = np.sign(dot_YW)

			if(class_sign != label[i]):
				err += 1

		return err/data.shape[0]




	def classify_A_VS_B(sf, a, b=None):

		if(b != None):
			#train
			labels_ind_a =  np.where(sf.input_label == a)[0]
			labels_ind_b =  np.where(sf.input_label == b)[0]

			#test
			labels_ind_a_T =  np.where(sf.test_label == a)[0]
			labels_ind_b_T =  np.where(sf.test_label == b)[0]
		
		if(b == None):
			#train
			labels_ind_a =  np.where(sf.input_label == a)[0]
			labels_ind_b =  np.where(sf.input_label != a)[0]

			#test
			labels_ind_a_T =  np.where(sf.test_label == a)[0]
			labels_ind_b_T =  np.where(sf.test_label != b)[0]
		
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

		train_data_label = np.hstack((data, label))

		np.random.shuffle(train_data_label)

		#print "data_label ", data_label
		data = train_data_label[:, :-1]
		#print "data ", data		
		label = train_data_label[:, -1]
		#print "label ", label
		
		data_test = np.vstack((sf.test_data[labels_ind_a_T,:],sf.test_data[labels_ind_b_T,:]))

		label_a = sf.test_label[labels_ind_a_T].reshape(len(labels_ind_a_T),1)
		label_a[:,0] = 1
		
		label_b = sf.test_label[labels_ind_b_T].reshape(len(labels_ind_b_T),1)		
		label_b[:,0] = -1

		label_test = np.vstack((label_a,label_b))

		return (data, label, data_test, label_test)


if __name__ == '__main__':

	ptrn = Perceptron()
	
	ptrn.run_one_vs_all()
