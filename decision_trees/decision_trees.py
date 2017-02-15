class decision_tree:
    def __init__(sf, filenames):
        sf._input_data = load(filenames[0])


    def load(sf,filename ):
        filehandle = open(filename, "r")
        line = filehandle.readline()
        while line != "":
            line = line.split()
            
            line = filehandle.readline()


    def train(sf):
        sf._rootnode = sf_input_data


    def information_gain(sf, X, Z):
        # uncertainty (entropy) reduced in data X when knowing Z
        return entropy(X) - conditional_entropy(X,Z)


    def entropy(sf,X):
        