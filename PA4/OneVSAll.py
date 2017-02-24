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
		
		sf.confusion_matrix = []
		sf.num_classes = 6

		con_mat_row = []
		for x in range(sf.num_classes):
			con_mat_row.append(0)

		for x in range(sf.num_classes+1):
			sf.confusion_matrix.append(list(con_mat_row))
		
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


	def run_one_vs_all(sf):

		#C1
		data1, label1, data_test1, label_test1 = sf.read_data(1)
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_1)

		#C2
		data2, label2, data_test2, label_test2 = sf.read_data(2)
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_2)

		#C3
		data3, label3, data_test3, label_test3 = sf.read_data(3)
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_3)
		
		#C4
		data4, label4, data_test4, label_test4 = sf.read_data(4)		
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_4)

		#C5
		data5, label5, data_test5, label_test5 = sf.read_data(5)
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_5)
		
		#C6
		data6, label6, data_test6, label_test6 = sf.read_data(6)
		
		sf.perceptron(data, label, data_test, label_test, sf.weight_mat_6)


	def test_one_vs_all(sf, data):
		
		predictions = []
		for x in range(data.shape[0]):
			class_sign = [0, 0, 0, 0, 0, 0]	
			#class 1
			dot_YW = np.dot(data[i], sf.weight_mat1)
			class_sign[0] = np.sign(dot_YW)

			#class 2
			dot_YW = np.dot(data[i], sf.weight_mat2)
			class_sign[1] = np.sign(dot_YW)

			#class 3
			dot_YW = np.dot(data[i], sf.weight_mat3)
			class_sign[2] = np.sign(dot_YW)

			#class 4
			dot_YW = np.dot(data[i], sf.weight_mat4)
			class_sign[3] = np.sign(dot_YW)

			#class 5
			dot_YW = np.dot(data[i], sf.weight_mat5)
			class_sign[4] = np.sign(dot_YW)

			#class 6
			dot_YW = np.dot(data[i], sf.weight_mat6)
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

			if dont_know == False:
				if no_class == False:
					label = class_label
				else:
					label =  -1
			else:
				label = -1 #dont know
			predictions += [label]
		return predictions
					return class_label
				else
					return -1
			else
				return -1 #dont know

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
	
	#ptrn.run_all_perceptron_algorithms()
	ptrn.run_one_vs_all()
	ptrn.test_one_vs_all()
