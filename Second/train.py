import numpy as np
import pickle
shape = (2,20,2)
model = pickle.load(open("model.txt","r"))
data = pickle.load(open("data.txt","r"))
inX = data["x"]
reg_lambda = 0.01
epsilon = 0.01
biases = model["biases"]
weights = model["weights"]
def sigmoid(val):
    return 1/(1+np.exp(-val))
def dsig(val):
    a = sigmoid(val)
    return a*(1-a)
# tries = input("Enter number of tries")
def calculate():
    inter=[]
    temp = inX
    inter.append(temp)
    for weight in weights:
        print weight.shape
        temp = np.dot(temp,weight)
        temp = sigmoid(temp)
        inter.append(temp)
    print "Inter"
    for i in inter:
        print i.shape

#
# model["weights"] = weights
# model["biases"] = biases
# pickle.dump(model,open("model.txt","w"))
if __name__ =="__main__":
    calculate()
