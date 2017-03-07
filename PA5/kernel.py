import numpy as np
from itertools import combinations

class Kernel:
    def __init__(sf):
        sf._sub_size = 0

        sf._input_data = []
        sf._input_label = []

        sf._test_data = []
        sf._test_label = []

        sf._weights = []

        sf.phi_func_data = []

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

    def set_weights(sf, sub_size): # sub_size = p (substring size))
        # Set the weights
        sf._sub_size = sub_size
        sf._weights = np.zeros(len(list(combinations(sf._alphabet, sub_size))))

    def train(sf, data, label, weights):
        for x in range(data.shape[0]):
            alphabet_count, string_count = sf.count_substrings(sf._alphabet,data[x,:][0],sf._sub_size)
            #k = K(np.array(string_count), np.array(x2_count))
            #print(string_count)
            if np.dot(weights.T,string_count) * label[x] <= 0:
                weights += (label[x] * string_count)
                #print(weights)

        return weights

    def predict(sf, data, label, weights):
        results = []
        for x in range(data.shape[0]):
            string_count, cur_string_count = sf.count_substrings(sf._alphabet,data[x,:][0],sf._sub_size)
            results += [np.sign(np.dot(weights.T,string_count))]
        
        return np.count_nonzero(np.array(results) == label)

    def count_substrings(sf, cur_string, string, p):
        substrings = list(combinations(cur_string, p))
        
        #cur_string = [s for s in list(combinations(alphaber,p)) if "".join(s) in alphaber]
        #string = [s for s in list(combinations(string,p)) if "".join(s) in string]
        
        cur_string_count = [] # Len substrings
        string_count = [] # Len substrings
        #print(substrings, cur_string, string)
        for sub in substrings:
            s = "".join(sub)

            cur_string_count += [sf.occurrences(cur_string,s)]
            string_count += [sf.occurrences(string,s)]
        #print(string_count)
        return np.array(alphabet_count), np.array(string_count)

    def occurrences(sf, string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count+=1
            else:
                return count

    def K(sf, phi_x, phi_x2):
        return sum(phi_x * phi_x2)

if __name__ == "__main__":
    DEFAULT_INPUT = "hw5train.txt"
    DEFAULT_TEST = "hw5test.txt"
    kernel = Kernel()
    d,l = kernel.read_data(DEFAULT_INPUT)
    print(d.shape, d.dtype, len(d[0][0]))
    print(l.shape, l.dtype, l[list(range(20))])
    kernel.set_weights(3)
    w = kernel.train(d,l, kernel._weights)
    r = kernel.predict(d,l,w)
    print(r)


