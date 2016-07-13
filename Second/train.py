import numpy as np
import pickle

shape = (2,10,10,2)
model = pickle.load(open("model.txt","rb"))
data = pickle.load(open("data.txt","rb"))
inX = data["x"]
inY = data["y"]
biases = model["biases"]
weights = model["weights"]

reg_lambda = 0.01
epsilon = 0.1

def sigmoidPlain(val):
    return 1/(1+np.exp(-val))
sigmoid = np.vectorize(sigmoidPlain)
def dsig(val):
    a = sigmoid(val)
    return a*(1-a)
# tries = input("Enter number of tries")
def calculate(inputValue):
    """
    Gives final value for some input
    """
    # inter=[]
    temp = inputValue
    # inter.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = sigmoid(temp)
        # inter.append(temp)
    return temp
def train(inputValue,actualValue):
    activ=[]
    temp = inputValue
    activ.append(temp)
    for weight in weights:
        temp = np.dot(temp,weight)
        temp = sigmoid(temp)
        activ.append(temp)
    #print(len(activ))
    deltaOut = temp - actualValue
    deltatemp = deltaOut.copy()
    backActiv = activ.copy()
    backActiv.reverse()
    backWeight = weights.copy()
    backWeight.reverse()
    deltaWeights = []
    for x in range(len(backWeight)):
        deltatemp = np.multiply(deltatemp,dsig(backActiv[x]))
        deltaWeights.append(deltatemp)
        deltatemp=np.dot(deltatemp,backWeight[x].T)
    deltaWeights.reverse()
    reducdDw = []
    for x in range(len(deltaWeights)):
        reducdDw.append(np.average(deltaWeights[x],axis=0))
    for x in range(len(weights)):
        biases[x]-=epsilon*reducdDw[x]
        wTemp = np.dot(activ[x].T,deltaWeights[x])
        weights[x]-=epsilon*wTemp
    nModel = {}
    nModel["biases"]=biases
    nModel["weights"]=weights
    pickle.dump(nModel,open("model.txt","wb"))

def cost(inputValue,actualValue):
    output = calculate(inputValue)
    #print(inputValue,output)
    temp = output - actualValue
    temp = np.power(temp,2)*0.5
    cost = np.sum(temp)
    cost = cost/len(temp)
    print(cost)
    return cost
def load():
    model = pickle.load(open("model.txt","rb"))
    data = pickle.load(open("data.txt","rb"))
    inX = data["x"]
    inY = data["y"]
    #model["weights"] = weights
    #model["biases"] = biases

if __name__ =="__main__":
    times = input("Times : ")
    for x in range(int(times)):
        cost(inX,inY)
        train(inX,inY)
        load()
