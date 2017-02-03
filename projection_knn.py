import numpy as np
from random import choice

def read_data(filename, projection = False):
    try:
        filehandle = open(filename,"r")
        file = filehandle.read()
        file = file.split()
        data = []
        if projection:
            for row in range(len(file)//20):
                data += [file[row*20:row*20+20]]
            P = np.array(data)
            return P.astype(float)
        else:
            for vector in range(len(file)//785):
                data += [file[vector*785:vector*785+785]]

            darray = np.array(data)
            X = darray[:,0:784]
            Y = darray[:,784]
            return X.astype(int), Y.astype(int)
        
    except IOError:
        print("IOError -- File does not exist")
        return


#X, Y = read_data("hw2train.txt")
#XTest, YTest = read_data("hw2validate.txt")


def knn(k, X, Y,XTest, num_classes):
    predictions = []
    for test_i in range(XTest.shape[0]):
        closest_classes = [-1]*k
        closest_dist = [-1]*k

        for i in range(X.shape[0]):
            dist = euclidean_dist(XTest[test_i],X[i])
            replace_ind = None

            if min(closest_dist) == -1:
                replace_ind = closest_dist.index(min(closest_dist))
            elif  max(closest_dist) > dist:
                replace_ind = closest_dist.index(max(closest_dist))
            if replace_ind != None:
                closest_dist[replace_ind] = dist
                closest_classes[replace_ind] = Y[i]
        class_count = [0]*num_classes
        for c in closest_classes:
            class_count[c] += 1
        # If tie, break randomly
        if class_count.count(max(class_count)) > 1:
            max_index = []
            for i in range(len(class_count)):
                if class_count[i] == max(class_count):
                    max_index += [i]
            c = choice(max_index)
        else :
            c = class_count.index(max(class_count))
        predictions += [c]
    return predictions
                    
        
def euclidean_dist(v1,v2):
    return np.sqrt(sum((v1-v2)**2))

def test():
    Y = np.array([1,1,1,1,1,2,2,2,2])
    X = np.array([[1,2],[1,0],[1,1],[0,0],[0,-1],[-1,0],[-1,-1],[-1,-2],[-2,-2]])
    XTest = np.array([[0,0],[1,1],[-1,-2]])
    YTest = np.array([1,1,2])
    results = (knn(3, X, Y, XTest, 10))
    print(results, YTest)

def t():
    X,Y = read_data("hw2train.txt")
    XTest, YTest = read_data("hw2validate.txt")
    results = knn(3, X,Y, X, 10)
    print(float((Y.shape[0] -( np.count_nonzero(np.array(results) ==  Y))))/Y.shape[0])

    print(results, Y)

#t()

def projection():

    	X,Y = read_data("hw2train.txt")
    	XValid, YValid = read_data("hw2validate.txt")
    	XTest, YTest = read_data("hw2test.txt")
	P = read_data("projection.txt", True)

	P_train = [np.dot(x, P) for x in X]
	P_valid = [np.dot(x, P) for x in XValid]
	P_test = [np.dot(x, P) for x in XTest]

	P_train = np.array(P_train)
	P_valid = np.array(P_valid)
	P_test = np.array(P_test)

	res1 = knn(1, P_train, Y, P_train, 10)
	res5 = knn(5, P_train, Y, P_train, 10)
	res3 = knn(3, P_train, Y, P_train, 10)
	res9 = knn(9, P_train, Y, P_train, 10)
	res15 = knn(15, P_train, Y, P_train, 10)
    	
	err_train = [0, 0, 0, 0, 0]

	err_train[0] = float((Y.shape[0] -( np.count_nonzero(np.array(res3) ==  Y))))/Y.shape[0]
	print "label 3: ", err_train[0]
	err_train[1] = float((Y.shape[0] -( np.count_nonzero(np.array(res5) ==  Y))))/Y.shape[0]
	print "label 3: ", err_train[1]
	err_train[2] = float((Y.shape[0] -( np.count_nonzero(np.array(res9) ==  Y))))/Y.shape[0]
	print "label 3: ", err_train[2]
	err_train[3] = float((Y.shape[0] -( np.count_nonzero(np.array(res15) ==  Y))))/Y.shape[0]
	print "label 3: ", err_train[3]
	err_train[4] = float((Y.shape[0] -( np.count_nonzero(np.array(res1) ==  Y))))/Y.shape[0]
	print "label 1: ", err_train[4]
	
	print "err_train: ", err_train
		
	#validation
	res1 = knn(1, P_train, Y, P_valid, 10)
	res5 = knn(5, P_train, Y, P_valid, 10)
	res3 = knn(3, P_train, Y, P_valid, 10)
	res9 = knn(9, P_train, Y, P_valid, 10)
	res15 = knn(15, P_train, Y, P_valid, 10)
	
	err = [0, 0, 0, 0]	
    	err[0] = float((YValid.shape[0] -( np.count_nonzero(np.array(res1) ==  YValid))))/YValid.shape[0]
    	err[1] = float((YValid.shape[0] -( np.count_nonzero(np.array(res5) ==  YValid))))/YValid.shape[0]
    	err[2] = float((YValid.shape[0] -( np.count_nonzero(np.array(res9) ==  YValid))))/YValid.shape[0]
    	err[3] = float((YValid.shape[0] -( np.count_nonzero(np.array(res15) ==  YValid))))/YValid.shape[0]

	minIdx = err.index(min(err))
	best_label = 0

	if(minIdx == 0):
		best_label = 1
	elif(minIdx == 0):
		best_label = 5
	elif(minIdx == 0):
		best_label = 9
	else:
		best_label = 12

	print "validation error: ", err[minIdx]

	res9 = knn(best_label, P_train, Y, P_test, 10)

    	err_test = float((YTest.shape[0] -( np.count_nonzero(np.array(res9) ==  YTest))))/YTest.shape[0]

	print "test error: ", err_test
	

projection()
