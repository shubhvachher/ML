import sklearn
import numpy as np
from time import time
np.random.seed(int(time()))
import sklearn.datasets as datasets
x,y = datasets.make_moons(200,noise=0.20)
dataset = {"x":x,"y":y}
import pickle
pickle.dump(dataset,open("data.txt","w"))
