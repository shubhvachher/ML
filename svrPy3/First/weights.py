dimension = (5,3)   #dimension(layers,nodes)
import numpy
import pickle
a = numpy.random.rand(dimension[0]-1,dimension[1],dimension[1])
a = a*10
weights = a-5
print(weights)
pickle.dump(weights,open("weights2.txt","wb"))
