from PIL import Image
from numpy import *
from pylab import *
import os, sift

def process_image_dsift(imagename,resultname,size=20,steps=10,
	force_orientation=False,resize=None):
	""" Process an image with densel sampled SIFT descriptors
	  and save the resuts in a file. Option input: size of feats
	  steps between locations, forcing computation of desc orientation
	  false means all are oriented up, tuple for resizing the image"""

	im = Image.open(imagename).convert('L')
	if resize!=None:
	  im = im.resize(resize)
	m,n = im.size

	if imagename[-3:] != 'pgm':
	  im.save('tmp.pgm')
	  imagename = 'tmp.pgm'

	scale = size/3.0
	x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
	xx,yy = x.flatten(),y.flatten()
	frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
	savetxt('tmp.frame',frame.T,fmt='%03.3f')

	if force_orientation:
	  cmmd = str("vlfeat/bin/glnxa64/sift "+imagename+" --output="+resultname+
	    " --read-frames=tmp.frame --orientations")
	else:
	  cmmd = str("vlfeat/bin/glnxa64/sift "+imagename+" --output="+resultname+
	    " --read-frames=tmp.frame")
	os.system(cmmd)
	print 'processed', imagename, 'to', resultname
