#!/usr/bin/python
__author__ = 'Joshua I. James'
import sys, getopt, os
from PIL import Image
from numpy import *
from pylab import *
import sift, imtools
from os.path import basename
import pickle
from scipy.cluster.vq import *
import pca, vocabulary

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

if img_dir != '':
  if os.path.isdir(img_dir):
	imlist = imtools.get_imlist(img_dir)
elif ifile != '':
  if os.path.isfile(ifile):
	imlist = list(ifile)

##########################################

imnbr = len(imlist)
for i in range(imnbr):
  im1 = array(Image.open(imlist[i]).convert('L'))
  siftName = basename(imlist[i])
  sift.process_image(imlist[i], "features/"+siftName+".sift")
  featlist.append("features/"+siftName+".sift")


voc = vocabulary.Vocabulary('kitty')
voc.train(featlist,1000,10)

#saving vocab
with open('codebooks/vocab.pkl', 'wb') as f:
  pickle.dump(voc,f)
print 'vocab is:', voc.name, voc.nbr_words
