from PIL import Image
from numpy import *
from pylab import *
import pca, imtools

imlist = imtools.get_imlist('corpus')

im = array(Image.open(imlist[0]))
m,n = im.shape[0:2]
imnbr = len(imlist)

for i in range(imnbr):
  tmpArr = [array(Image.open(imlist[i])).flatten()]

immatrix = array(tmpArr, 'f')

V,S,immean = pca.pca(immatrix)


