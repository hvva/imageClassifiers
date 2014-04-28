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

# Store input and output file names
ifile=''
ofile=''
img_dir=''
imlist=''
featlist = []
# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:o:d:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-i':
        ifile=a
    elif o == '-o':
        ofile=a
    elif o == '-d':
        img_dir=a
    else:
        print("Usage: %s -d directory [-i image] -o output" % sys.argv[0])

#if img_dir != '':
#  if os.path.isdir(img_dir):
#	imlist = imtools.get_imlist(img_dir)
#elif ifile != '':
#  if os.path.isfile(ifile):
#	imlist = list(ifile)

##########################################

imlist = imtools.get_imlist('corpus/train/')
imnbr = len(imlist)
for i in range(imnbr):
  im1 = array(Image.open(imlist[i]).convert('L'))
  siftName = basename(imlist[i])
  dsift.process_image_dsift(imlist[i], "features/train/"+siftName+".dsift",10,5,resize=(50,50))
  #featlist.append("features/"+siftName+".dsift")

imlist = imtools.get_imlist('corpus/test/')
imnbr = len(imlist)
for i in range(imnbr):
  im1 = array(Image.open(imlist[i]).convert('L'))
  siftName = basename(imlist[i])
  dsift.process_image_dsift(imlist[i], "features/test/"+siftName+".dsift",10,5,resize=(50,50))

def read_feature_labels(path):
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	features = []
	for featfile in featlist:
	  l,d = sift.read_features_from_file(featfile)	
	  features.append(d.flatten())
	features = array(features)

	# create labels
	labels = [featfile.split('/')[-1][0] for featfile in featlist]

	return features,array(labels)


features,labels = read_feature_labels('features/train/')
test_features,test_labels = read_feature_labels('features/test/')
classnames = unique(labels)

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
param = svm_parameter('-t 0')

m = svm_train(prob,param)

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

