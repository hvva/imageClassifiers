from PIL import Image
import os

def check_image_with_pil(path):
	""" Send a full path to an image to verify it is correct """
	try:
		Image.open(path)
	except IOError:
		return False
	return True

def get_imlist(path):
	""" Returns a list of filenames for all valid images in a dir """
	imlist = []
	for f in os.listdir(path):
	  if check_image_with_pil(os.path.join(path,f)):
	    imlist.append(os.path.join(path,f))

	return imlist	
	
	#return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im,sz):
	""" Resize an image array using PIL """
	pil_im = Image.fromarray(uint8(im))

	return array(pil_im.resize(sz))

def histeq(im,nbr_bins=256):
	""" Histogram equalization of a grayscale image """

	# get image histogram
	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum()
	cdf = 255 * cdf / cdf[-1] # normalize

	# user linear interpolation of cdf to find new pixel values
	im2 = interp(im.flatten(),bins[:-1],cdf)

	return im2.reshape(im.shape), cdf

def compute_average(imlist):
	""" Compute the average of a list of images """
	
	# open first image and make into array of type float
	averageim = array(Image.open(imlist[0]), 'f')

	for imname in imlist[1:]:
	  try:
	    averageim += array(Image.open(imname))
	  except:
	    print imname + '...skipped'
	averageim /= len(imlist)

	# return average as uint8
	return array(averageim, 'uint8')

def plot_2D_boundry(plot_range,points,decisionfcn,labels,values=[0]):
  """ Plot range is (xmin,xmax,ymin,ymax) points is a list
    of class points, decisionfcn is a function to evaluate
    labels is a list of labels that decisionfcn returns for each class
    values is a list of decision contrours to show """

  clist = ['b','r','g','k','m','y']

  x = arange(plot_range[0],plot_range[1],.1)
  y = arange(plot_range[2],plot_range[3],.1)
  xx,yy = meshgrid(x,y)
  xxx,yyy = xx.flatten(),yy.flatten()
  zz = array(decisionfcn(xxx,yyy))
  zz = zz.reshape(xx.shape)
  contour(xx,yy,zz,values)

  for i in range(len(points)):
    d = decisionfcn(points[i][:,0],points[i][:,1])
    correct_ndx = labels[i]==d
    incorrect_ndx = labels[i]!=d
    plot(points[i][correct_ndx,0],points[i][correct_ndx,1],'*',color=clist[i])
    plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],'o',color-clist[i])

  axist('equal')
