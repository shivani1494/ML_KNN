import numpy as np
from itertools import combinations

class Kernel:
    def __init__(sf):
        sf._sub_size = 0 # Set when calling train

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

        print(filename,"loaded : Dim",data.shape)
        return data[:,:-1], data[:,-1].astype(int)

    def train(sf, data, label, p, kernel):
        weights = []
        sf._sub_size = p
        for x in range(data.shape[0]):
            print(x)

            #print(string_count)
            if label[x]*kernel(data[x,:][0], weights) <= 0:
                weights += [[data[x,:][0],label[x]]]
                print("num Weights:",len(weights))
                #print(weights)

        return weights

    def predict(sf, data, label, weights, kernel):
        results = []
        for x in range(data.shape[0]):
            results += [np.sign(kernel(data[x,:][0],weights))]
        
        return ((data.shape[0] - np.count_nonzero(np.array(results) == label))/data.shape[0])

    def kernelizePerceptron(sf, cur_string, weights):
        k = 0
        for w in weights:
            cur_string_count, string_count = sf.count_substrings(cur_string, w[0],sf._sub_size)
            k += w[1]*np.dot(cur_string_count.T, string_count)
        return k

    def kernelizePerceptron_one_ocurrence(sf, cur_string, weights):
        k = 0
        for w in weights:
            cur_string_count, string_count = sf.count_substrings(cur_string, w[0],sf._sub_size)
            k += w[1] * np.count_nonzero(cur_string_count * string_count)
        return k


    def count_substrings(sf, cur_string, string, p):
        substrings = ["".join(s) for s in set(list(combinations(cur_string,p))) if "".join(s) in cur_string]
        
        #cur_string = [s for s in list(combinations(alphaber,p)) if "".join(s) in alphaber]
        #string = [s for s in list(combinations(string,p)) if "".join(s) in string]
        
        cur_string_count = [] # Len substrings
        string_count = [] # Len substrings
        #print(substrings, cur_string, string)
        for sub in substrings:

            cur_string_count += [sf.occurrences(cur_string,sub)]
            string_count += [sf.occurrences(string,sub)]
        #print(string_count)
        return np.array(cur_string_count), np.array(string_count)

    def occurrences(sf, string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count+=1
            else:
                return count

if __name__ == "__main__":
    
    # Files
    DEFAULT_INPUT = "hw5train.txt"
    DEFAULT_TEST = "hw5test.txt"
    
    # Instantiate Kernel
    kernel = Kernel()

    # Read Training and Testing Data
    input_data, input_label = kernel.read_data(DEFAULT_INPUT)
    test_data, test_label = kernel.read_data(DEFAULT_TEST)
    #print(kernel.kernelizePerceptron("bce",[["abc",-1],["cdf",1]]))
    
    # Training Part 1...

    for p in [3,4,5]:
        print("************* p = ",p)
        output = open("tests.txt","a")
        output.write("\nP: %i" %(p))

        weights = kernel.train(input_data,input_label, p, kernel.kernelizePerceptron)
        
        error = kernel.predict(input_data,input_label,weights, kernel.kernelizePerceptron)
        print("Training Error:",error)
        output.write("\nTraining Error: %f" %(error))
        
        error_test = kernel.predict(test_data,test_label,weights, kernel.kernelizePerceptron)
        print("Testing Error:",error_test)
        output.write("\nTest Error: %f" %(error_test))
        
        output.close()

    # Training Part 2...

    for p in [3,4,5]:
        print("************* p = ",p)
        output = open("tests.txt","a")
        output.write("\nP: %i" %(p))

        weights = kernel.train(input_data,input_label, p, kernel.kernelizePerceptron_one_ocurrence)
        
        error = kernel.predict(input_data,input_label,weights, kernel.kernelizePerceptron_one_ocurrence)
        print("Training Error:",error)
        output.write("\nTraining Error: %f" %(error))
        
        error_test = kernel.predict(test_data,test_label,weights, kernel.kernelizePerceptron_one_ocurrence)
        print("Testing Error:",error_test)
        output.write("\nTest Error: %f" %(error_test))

        output.close()