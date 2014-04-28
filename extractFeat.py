#!/usr/bin/python
import sys, getopt, os
from PIL import Image
from numpy import *
from pylab import *
import sift, dsift, imtools
from os.path import basename

typeFeats='dsift'
dirImage=''
dirFeats=''
imgSize='50'
featTypes=['sift','dsift']

def print_error():
  print("Usage: %s -f feat_type -i image_dir -o feat_dir -r imgSize" % sys.argv[0])
  print("    -f: Feature types [sift/dsift]")
  print("    -i: Directory containing images")
  print("    -o: Directory to store features")
  print("    -r: Size to resize images ex: 50")
  sys.exit(1)

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"f:i:o:r:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-f':
        typeFeats=a
    elif o == '-i':
        dirImage=a
    elif o == '-o':
        dirFeats=a
    elif o == '-r':
	imgSize=a
    else:
        print_error()


if not typeFeats or typeFeats not in featTypes:
  print "No/incorrect feature type selected"
  print_error()

if not dirImage or not os.path.isdir(dirImage):
  print "No/invalid image directory specified"
  print_error()

if not dirFeats or not os.path.isdir(dirFeats):
  print "No/invalid feature directory specified"
  print_error()

imgSize=int(imgSize)

if typeFeats == 'sift':
  imlist = imtools.get_imlist(dirImage)
  imnbr = len(imlist)
  for i in range(imnbr):
    im1 = array(Image.open(imlist[i]).convert('L'))
    siftName = basename(imlist[i])
    sift.process_image(imlist[i], dirFeats+siftName+".sift",params="--edge-thresh 10 --peak-thresh 5", resize=(imgSize,imgSize))
    #featlist.append("features/"+siftName+".sift")

if typeFeats == 'dsift':
  imlist = imtools.get_imlist(dirImage)
  imnbr = len(imlist)
  for i in range(imnbr):
    im1 = array(Image.open(imlist[i]).convert('L'))
    siftName = basename(imlist[i])
    dsift.process_image_dsift(imlist[i], dirFeats+siftName+".dsift",10,5,resize=(imgSize,imgSize))
    #featlist.append("features/"+siftName+".dsift")

