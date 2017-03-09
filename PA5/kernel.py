import numpy as np
import pickle

class Kernel:
    def __init__(sf):
        sf._sub_size = 0  # Set when calling train

    def read_data(sf, filename):
        # read test train and label data

        print("Loading %s ..." % filename)

        filehandle = open(filename, "r")
        line = filehandle.readline()
        data = []

        while line != "":
            line = line.split()
            data += [line]
            line = filehandle.readline()

        data = np.array(data)

        print(filename, "loaded : Dim", data.shape)
        return data[:, :-1], data[:, -1].astype(int)

    def train(sf, data, label, p, kernel):
        weights = []
        sf._sub_size = p
        for x in range(data.shape[0]):
            print(x)

            # print(string_count)
            if label[x] * kernel(data[x, :][0], weights, data, label) <= 0:
                weights += [x]
                print("num Weights:", len(weights))
                # print(weights)

        return weights

    def predict(sf, data, label, weights_data, weights_label, weights, kernel):
        results = []
        print("Testing...")
        for x in range(data.shape[0]):
            print(x)
            results += [np.sign(kernel(data[x, :][0], weights, weights_data, weights_label))]

        return ((data.shape[0] - np.count_nonzero(np.array(results) == label)) / data.shape[0])

    def kernelizePerceptron(sf, cur_string, weights, weights_data, weights_label):
        p = sf._sub_size
        substrings = [cur_string[s:s + p] for s in range(len(cur_string) - p + 1)]
        substrings = set(substrings)
        cur_string_count = []
        for sub in substrings:
            cur_string_count += [sf.occurrences(cur_string, sub)]
        cur_string_count = np.array(cur_string_count)

        k = 0
        for w in weights:
            string_count = sf.count_substrings(substrings, weights_data[w][0])
            k += weights_label[w] * np.dot(cur_string_count.T, string_count)
        return k

    def kernelizePerceptron_one_ocurrence(sf, cur_string, weights, weights_data, weights_label):
        p = sf._sub_size
        substrings = [cur_string[s:s + p] for s in range(len(cur_string) - p + 1)]
        substrings = set(substrings)
        cur_string_count = []
        for sub in substrings:
            cur_string_count += [sf.occurrences(cur_string, sub)]
        cur_string_count = np.array(cur_string_count)

        k = 0
        for w in weights:
            string_count = sf.count_substrings(substrings, weights_data[w][0])
            k += weights_label[w] * np.count_nonzero(cur_string_count * string_count)
        return k

    def count_substrings(sf, substrings, string):
        string_count = []  # Len substrings
        # print(substrings, cur_string, string)
        for sub in substrings:
            string_count += [sf.occurrences(string, sub)]
        # print(string_count)
        return np.array(string_count)

    def occurrences(sf, string, sub):
        count = start = 0
        while True:
            start = string.find(sub, start) + 1
            if start > 0:
                count += 1
            else:
                return count

    def max_coordinates(sf, data, label, weights, num_max=2, p=5):
        global_substrings = {}

        for w in weights:
            substrings = [data[w][0][s:s + p] for s in range(len(data[w][0]) - p + 1)]
            for sub in substrings:
                if sub in global_substrings:
                    global_substrings[sub] += 1 * label[w]
                else:
                    global_substrings[sub] = 1 * label[w]

        print(global_substrings)

        max_coordinates = []
        for m in range(num_max):
            key = max(global_substrings, key=global_substrings.get)
            max_coordinates += [[key,global_substrings[key]]]
            global_substrings[key] = -1 # Remove this substring as the max so next time max is 2nd substring

        return max_coordinates

    def max_coordinates_kernel(sf,num_max, data, weights, p=5):
        global_substrings = {}
        for x in range(data.shape[0]):
            substrings = [data[x][0][s:s + p] for s in range(len(data[x][0]) - p + 1)]
            substrings = set(substrings)
            cur_string_count = sf.count_substrings_dic(substrings,data[x][0])
            for w in weights:
                string_count = sf.count_substrings_dic(substrings,w[0])
                for sub in substrings:
                    if sub in string_count:
                        if sub in global_substrings:
                            global_substrings[sub] += cur_string_count[sub] * string_count[sub]
                        else:
                            global_substrings[sub] = cur_string_count[sub] * string_count[sub]


        max_coordinates = []
        for m in range(num_max):
            key = max(global_substrings, key=global_substrings.get)
            max_coordinates += [[key,global_substrings[key]]]
            global_substrings[key] = -1 # Remove this substring as the max so next time max is 2nd substring

        return max_coordinates

    def count_substrings_dic(sf, substrings, string):
        string_count = {}  # Len substrings
        for sub in substrings:
            string_count[sub] = sf.occurrences(string, sub)
        return string_count



if __name__ == "__main__":
    # Files
    DEFAULT_INPUT = "hw5train.txt"
    DEFAULT_TEST = "hw5test.txt"

    # Instantiate Kernel
    kernel = Kernel()

    # Read Training and Testing Data
    input_data, input_label = kernel.read_data(DEFAULT_INPUT)
    test_data, test_label = kernel.read_data(DEFAULT_TEST)
    # print(kernel.kernelizePerceptron("bce",[["abc",-1],["cdf",1]]))

    #weights = kernel.train(input_data, input_label, 2, kernel.kernelizePerceptron)
    #error = kernel.predict(input_data, input_label, input_data, input_label, weights, kernel.kernelizePerceptron)
    #print(error)

    # Part 3
    #weights = kernel.train(input_data, input_label, 5, kernel.kernelizePerceptron)

    varFile = open("objs.pickle","rb")
    #pickle.dump([weights],varFile)
    weights = pickle.load(varFile)[0]
    print(len(weights))
    varFile.close()

    #####kernel._sub_size = 5
    #####error = kernel.predict(input_data, input_label, input_data, input_label, weights, kernel.kernelizePerceptron)
    #####print("Training Error:", error)

    max = kernel.max_coordinates(input_data, input_label,weights,num_max=6)
    print(max)

    # Training Part 1...

    for p in [3,4,5]:
        print("************* p = ",p)
        output = open("tests.txt","a")
        output.write("\nP: %i" %(p))

        weights = kernel.train(input_data,input_label, p, kernel.kernelizePerceptron)
        
        error = kernel.predict(input_data,input_label, input_data, input_label, weights, kernel.kernelizePerceptron)
        print("Training Error:",error)
        output.write("\nTraining Error: %f" %(error))
        
        error_test = kernel.predict(test_data,test_label, input_data, input_label,weights, kernel.kernelizePerceptron)
        print("Testing Error:",error_test)
        output.write("\nTest Error: %f" %(error_test))
        
        output.close()

    # Training Part 2...

    for p in [3,4,5]:
        print("************* p = ",p)
        output = open("tests.txt","a")
        output.write("\nP: %i" %(p))

        weights = kernel.train(input_data,input_label, p, kernel.kernelizePerceptron_one_ocurrence)
        
        error = kernel.predict(input_data,input_label, input_data, input_label, weights, kernel.kernelizePerceptron_one_ocurrence)
        print("Training Error:",error)
        output.write("\nTraining Error: %f" %(error))
        
        error_test = kernel.predict(test_data,test_label, input_data, input_label, weights, kernel.kernelizePerceptron_one_ocurrence)
        print("Testing Error:",error_test)
        output.write("\nTest Error: %f" %(error_test))

        output.close()
