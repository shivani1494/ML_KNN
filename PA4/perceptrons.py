import numpy as np
import copy

### DEFAULT FILENAMES
    TRAINING_NAME = "hw4train.txt"
    TEST_NAME = "hw4test.txt" 
    DICTIONARY_NAME = "hw4dictionary.txt"

class Perceptron:


    #make sure everything is in numpy
    def __init__(sf):

        sf.input_data = sf.read_data(TRAINING_NAME)

        sf.test_data = sf.read_data(TEST_NAME)

        #initialize to all 0's
        sf.weight_mat = []
        for i in range(sf.input_data.shape[1]):
            sf.weight_mat.append(0.0)
        sf.weight_mat = np.array(sf.weight_mat)

        #save weigth count and weights for voted and averaged perceptrons
        sf.weight_count = []
        sf.all_weight_mat = []

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

    def perceptron(sf):

        num_passes = 2

        #is the hyperplane getting updated every time
        #we compute a new weight vector?
        #such that the weigth vector is always normla to the 
        #hyperplane?

        for t in range(num_passes):         
            for i in range(sf.input_data.shape[0]):
                dot_XW = np.dot(sf.input_data[i], sf.weight_mat)
                if(sf.label[i]*dot_XW <= 0):
                    sf.weight_mat = sf.weight_mat + (sf.label[i]*sf.input_data[i])

            sf.train_err += [sf.run_perceptron(sf.input_data)] 
            print("train err after", t+1 ,"pass:", sf.train_err[-1])
            sf.test_err +=  [sf.run_perceptron(sf.test_data)]
            print("test err after", t+1, "pass:", sf.test_err[-1])


    def run_perceptron(sf, data):
        #predict output for normal perceptron

        err = 0.0

        for i in range(data.shape[0]):
            dot_YW = np.dot(sf.test_data[i], sf.weight_mat)
            if(dot_YW < 0 and label_output[i] == 1):
                err += 1
            elif(dot_YW > 0 and label_output[i] == -1):
                err += 1

        return err/data.shape[0]


    def voted_perceptron(sf):
        count = 1
        num_passes = 3

        for t in range(num_passes):        
            for i in range(len(sf.input_data)):
                dot_XW = np.dot(sf.input_data[i], sf.weight_mat)
                if(sf.label[i]*dot_XW <= 0):
                    temp_weight_mat = sf.weight_mat + (sf.label[i]*sf.input_data[i])
                    
                    #append for the previous matrix
                    sf.all_weight_mat.append(copy.copy(sf.weight_mat))
                    sf.weight_count.append(count)

                    sf.weight_mat = temp_weight_mat 
                    count = 1
                    
                else:
                    count += 1

            sf.run_voted_perceptron(t)

            sf.train_err += [run_voted_perceptron(sf.input_data)]
            print("train err after", t+1, "pass:", sf.train_err[t])
            sf.test_err += [run_voted_perceptron(sf.test_data)]
            print("train err after", t+1, "pass:", sf.test_err[t])




    def run_voted_perceptron(sf, data):

        sum_sign = 0
        for t in range(len(test_data)):

            #you can totally vectorize this loop
            for i in range(len(sf.all_weight_mat)):

                dot_WY = np.dot(sf.all_weight_mat[i]*test_data[t])
                sum_sign += sf.weight_count[i]*np.sign(dot_WY)

                #final sign or class of test data t
            class_t = np.sign(sum_sign)

            if(class_t != label_test[t]):
                err += 1

       return err/data.shape[0]


    #think about why would voted and averaged perceptron give you the
    #same result?!
    def averaged_perceptron(sf):
        pass



if __name__ == '__main__':

    ptrn = Perceptron()
    X, Y = ptrn.read_data(TRAINING_NAME)
    print(X.shape,Y.shape)
    #ptrn.perceptron()
    #ptrn.voted_perceptron()
    #ptrn.averaged_perceptron()
