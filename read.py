import numpy as np

def read_data(filename):
    try:
        filehandle = open(filename,"r")
        file = filehandle.read()
        file = file.split()
        data = []
        for vector in range(len(file)//785 - 1):
            data += [file[vector*785:vector*785+785+1]]

        darray = np.array(data)
        X = darray[:,0:784]
        Y = darray[:,784]
        return X, Y
        
    except IOError:
        print("IOError -- File does not exist")
        return

#X, Y = read_data("hw2train.txt")
#XTest, YTest = read_data("hw2validate.txt")

