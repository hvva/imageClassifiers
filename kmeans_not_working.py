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
import pca

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

im = array(Image.open(imlist[0]))
m,n = im.shape[0:2]
imnbr = len(imlist)

for i in range(imnbr):
  tmpArr = [array(Image.open(imlist[i])).flatten()]

immatrix = array(tmpArr,'f')

V,S,immean = pca.pca(immatrix)

immean = immean.flatten()
projected = array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

projected = whiten(projected)
centroids,distortion = kmeans(projected,2)

code,distance = vq(projected,centroids)

for k in range(2):
  ind = where(code==k)[0]
  figure()
  gray()
  for i in range(minimum(len(ind),40)):
    subplot(2,10,i+1)
    imshow(immatrix[ind[i]].reshape((25,25)))
    axis('off')
show()
