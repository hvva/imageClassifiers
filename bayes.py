from PIL import Image
from numpy import *
from pylab import *

class BayesClassifier(object):

	def __init__(self):
	  """ Initalize classifer with training data """

	  self.labels = []
	  self.mean = []
	  self.var = []
	  self.n = 0

	def train(self,data,labels=None):
	  """ Train on data (list of arrays n*dim)
	    Labels are optional, default is 0...n-1 """

	  if labels==None:
	    labels = range(len(data))
	  self.labels = labels
	  self.n = len(labels)

	  for c in data:
	    self.mean.append(mean(c,axis=0))
	    self.var.append(var(c,axis=0))

	def classify(self,points):
	  """  Classify the points by computing probs
	    for each class and return most probably label """

	  est_prob = array([gauss(m,v,points) for m,v in zip(self.mean,self.var)])

	  ndx = est_prob.argmax(axis=0)
	  est_labels = array([self.labels[n] for n in ndx])

	  return est_labels, est_prob

def gauss(m,v,x):
  """ evaluate gaussian in d-diensions with independent
    mean m and variance v at the points in (the rows of) x """

  if len(x.shape)==1:
    n,d = 1,x.shape[0]
  else:
    n,d = x.shape

  S = diag(1/v)
  x = x-m
  y = exp(-0,5*diag(dot(x,dot(S,x.T))))

  return y * (2*pi)**(-d/2.0) / ( sqrt(prod(v)) + 1e-6)
