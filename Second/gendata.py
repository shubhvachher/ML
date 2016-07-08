import sklearn
import numpy as np
from time import time
np.random.seed(int(time()))
import sklearn.datasets as datasets
x,y = datasets.make_moons(200,noise=0.20)
actY = np.zeros([200,2])
actY[range(200),y]=1
dataset = {"x":x,"y":actY}
import pickle
pickle.dump(dataset,open("data.txt","w"))
