#!/usr/bin/python
__author__ = 'Joshua I. James'
import sys, getopt, os, pca
from PIL import Image
from numpy import *
from pylab import *
import sift, dsift, imtools
from os.path import basename
import pickle
from scipy.cluster.vq import *
from svmutil import *
import sift

# Store input and output file names
# Set if you want a default
typeFeats='dsift'
featTypes=['sift','dsift']
dirTrain='features/train'
dirTest='features/test'
use_pca=''
imlist=''
featlist = []
svm_params="-t 0"

def print_error():
  print("Usage: %s -f feat_type -l train_dir -t test_dir -s svm_params -p 1" % sys.argv[0])
  print("    -f: Feature types [sift/dsift]")
  print("    -l: Directory of training features")
  print("    -t: Directory of test features")
  print("    -s: libsvm paramaters - quoted")
  print("    -p: Use PCA")
  sys.exit(1)

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"f:l:t:p:s:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-f':
        typeFeats=a
    elif o == '-l':
        dirTrain=a
    elif o == '-t':
        dirTest=a
    elif o == '-s':
        svm_params=a
    elif o == '-p':
        use_pca='1'
    else:
        print_error()

if not typeFeats or typeFeats not in featTypes:
  print "No/incorrect feature type selected"
  print_error()

if not dirTrain or not os.path.isdir(dirTrain):
  print "No/invalid training directory specified"
  print_error()

if not dirTest or not os.path.isdir(dirTest):
  print "No/invalid test directory specified"
  print_error()

def read_feature_labels(path):
	if typeFeats == 'dsift':
		featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
	elif typeFeats == 'sift':
		featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.sift')]
	else:
		print_error()
	
	features = []
	for featfile in featlist:
	  l,d = sift.read_features_from_file(featfile)	
	  features.append(d.flatten())
	features = array(features)

	# create labels
	labels = [featfile.split('/')[-1][0] for featfile in featlist]

	return features,array(labels)


features,labels = read_feature_labels(dirTrain)
test_features,test_labels = read_feature_labels(dirTest)
classnames = unique(labels)

if use_pca:
  V,S,m = pca.pca(features)
  V = V[:50]
  features = array([dot(V,f-m) for f in features])
  test_features = array([dot(V,f-m) for f in test_features])

features = map(list,features)
test_features = map(list,test_features)

transl = {}
for i,c in enumerate(classnames):
  transl[c],transl[i] = i,c 

def convert_labels(keys,dict):
	""" keys should be an array of keys """
	newKeys = []
	for index, key in enumerate(keys):
	  newKeys.append(dict[key])
	
	return newKeys

prob = svm_problem(convert_labels(labels,transl),features)
param = svm_parameter(svm_params)

m = svm_train(prob,param)

#svm_save_model('libsvm.model',m)
#m = svm_load_model('libsvm.model')

res = svm_predict(convert_labels(labels,transl),features,m)

res = svm_predict(convert_labels(test_labels,transl),test_features,m)[0]
res = convert_labels(res,transl)

# accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print 'Accuracy:', acc

def print_confusion(res,labels,classnames):

  n = len(classnames)

  class_ind = dict([(classnames[i],i) for i in range(n)])

  confuse = zeros((n,n))
  for i in range(len(test_labels)):
    confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1

  print 'Confusion matrix for'
  print classnames
  print confuse


print_confusion(res,test_labels,classnames)

