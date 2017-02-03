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
            for vector in range(len(file)//785 -1):
                data += [file[vector*785:vector*785+785+1]]

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

t()
