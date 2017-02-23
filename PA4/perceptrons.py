import numpy as np

class Perceptron:

    #make sure everything is in numpy
    def init(sf):

        sf.input_data = []

        sf.input_data = np.array(sf.input_data)

        sf.label_input = [] #?

        sf.label_input = np.array(sf.label_input)

        sf.test_data = []

        sf.test_data = np.array(sf.test_data)

        #initialize to all 0's
        sf.weight_mat = []
        for i in range(len(input_data[0])):
            sf.weight_mat.append(0.0)

        sf.weight_count = []
        sf.all_weight_mat = []

        sf.weight_mat = np.array(sf.weight_mat)

        sf.train_err = [0.0, 0.0, 0.0]

        sf.test_err = [0.0, 0.0, 0.0]

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

        err = 0.0

        #is the hyperplane getting updated every time
        #we compute a new weight vector?
        #such that the weigth vector is always normla to the 
        #hyperplane?

        for t in xrange(3):         
            for i in range(len(sf.input_data)):
                dot_XW = np.dot(sf.input_data[i], sf.weight_mat)
                if(sf.label[i]*dot_XW <= 0):
                    sf.weight_mat = sf.weight_mat + (sf.label[i]*sf.input_data[i])
                    err += err

            sf.train_err[t] = sf.calculate_training_error(err)
            print("train err after", t+1 ,"pass:", sf.train_err[t])
            err = 0.0
            sf.run_perceptron(t)

    def run_perceptron(sf, t):

        err = 0.0

        for i in range(len(sf.test_data)):
            dot_YW = np.dot(sf.test_data[i], sf.weight_mat)
            if(dot_YW < 0 and label_output[i] == 1):
                err += err
            elif(dot_YW > 0 and label_output[i] == -1):
                err += 1

        sf.test_err[t] = sf.calculate_testing_error(err)
        print("train err after", t+1, "pass:", sf.test_err[t])

    def calculate_training_error(sf, err):
        return err/len(sf.input_data)


    def calculate_testing_error(sf, err):
        return err/len(sf.test_data)


    def voted_perceptron(sf):
        err = 0.0
        count = 1

        for t in xrange(3):         
            for i in range(len(sf.input_data)):
                dot_XW = np.dot(sf.input_data[i], sf.weight_mat)
                if(sf.label[i]*dot_XW <= 0):
                    temp_weight_mat = sf.weight_mat + (sf.label[i]*sf.input_data[i])
                    
                    #append for the previous matrix
                    sf.all_weight_mat.append(list(sf.weight_mat))
                    sf.weight_count.append(count)

                    sf.weight_mat = temp_weight_mat 
                    count = 1

                    err += 1                    
                else:
                    count += 1

            sf.run_voted_perceptron(t)

            sf.train_err[t] = sf.calculate_training_error(err)
            print("train err after", t+1, "pass:", sf.train_err[t])
            err = 0.0



    def run_voted_perceptron(sf, t):

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

        sf.test_err[t] = sf.calculate_testing_error(err)
        print("train err after", t+1, "pass:", sf.test_err[t])



    #think about why would voted and averaged perceptron give you the
    #same result?!
    def averaged_perceptron(sf):
        pass



if __name__ == '__main__':
    ### DEFAULT FILENAMES
    TRAINING_NAME = "hw4train.txt"
    TEST_NAME = "hw4test.txt" 
    DICTIONARY_NAME = "hw4dictionary.txt"

    ptrn = Perceptron()
    X, Y = ptrn.read_data(TRAINING_NAME)
    print(X.shape,Y.shape)
    #ptrn.perceptron()
    #ptrn.voted_perceptron()
    #ptrn.averaged_perceptron()


