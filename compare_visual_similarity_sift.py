#!/usr/bin/python
__author__ = 'Joshua I. James'
import sys, getopt, os
from PIL import Image
from numpy import *
from pylab import *
import sift, imtools
from os.path import basename

# Store input and output file names
ifile=''
ofile=''
img_dir=''
imlist=''
featlist = []

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"d:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-d':
        img_dir=a
    else:
        print("Usage: %s -d directory" % sys.argv[0])

if img_dir != '':
  if os.path.isdir(img_dir):
	imlist = imtools.get_imlist(img_dir)
elif ifile != '':
  if os.path.isfile(ifile):
	imlist = list(ifile)

##########################################

nbr_images = len(imlist)

# Get SIFT Features for each image
for i in range(nbr_images):
  im1 = array(Image.open(imlist[i]).convert('L'))
  siftName = basename(imlist[i])
  sift.process_image(imlist[i], "features/"+siftName+".sift")
  featlist.append("features/"+siftName+".sift")

matchscores = zeros((nbr_images,nbr_images))
for i in range(nbr_images):
  for j in range(i,nbr_images):
    if imlist[i] != imlist[j]:
      print 'comparing ', imlist[i], imlist[j]

      l1,d1 = sift.read_features_from_file(featlist[i])
      l2,d2 = sift.read_features_from_file(featlist[j])

      matches = sift.match_twosided(d1,d2)

      nbr_matches = sum(matches > 0)
      print 'number of matches = ', nbr_matches
      matchscores[i,j] = nbr_matches

for i in range(nbr_images):
  for j in range(i+1,nbr_images):
    matchscores[j,i] = matchscores[i,j]
