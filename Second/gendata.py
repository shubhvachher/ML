"""Generates data in the form of a dictionary with x and y.
shape of x as (200,2), 2 numbers based on make_moons from sklearn
shape of y as (200,2) multiclass labels for x"""
import numpy as np
from time import time
np.random.seed(int(time())) # is this for make_moons?
import sklearn.datasets as datasets
x,y = datasets.make_moons(200,noise=0.20)
actY = np.zeros([200,2])
actY[range(200),y]=1
dataset = {"x":x,"y":actY}
import pickle
pickle.dump(dataset,open("data.txt","wb"))
