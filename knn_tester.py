#!/usr/bin/python
#tested on windows 7x64, python 2.7.5
__author__ = 'JaeHun Jeong'

import re, os, sys, time, shlex, signal, shutil, random, imtools, subprocess, getopt

n = 5
typeFeats = 'dsift'
modelOut = None
dirCps = 'corpus/'
dirFeat = 'features/'
dirMan = dirCps+'man-1408860850/'
dirWoman = dirCps+'woman-1408859780/'
dirChild = dirCps+'child-1408860955/'
py_classifier = 'knn_classifier_model2.py'
py_extractfeat = 'extractFeat.py'
global_counter = 0
global_maxcount = None
global_average = None
global_accurate = 0
global_kimglist = None
global_nkimglist = None

def print_error():
	print("Usage: %s [-n number] [-c count] [-m model_out]" % sys.argv[0])
	print("		-n: Number of train images per labels (default = "+str(n)+")")
	print("		-c: maxcount (default = unlimited)")
	print("		-m: Model output file (default = "+str(modelOut)+")")
	sys.exit(1)

myopts, args = getopt.getopt(sys.argv[1:],"n:c:m:")
for o, a in myopts:
		if o == '-n':
				n = int(a)
		if o == '-m':
				modelOut = a
		if o == '-c':
				global_maxcount = int(a)

if len(sys.argv)<=1:
	print_error()

if global_maxcount and global_maxcount<=0:
	print "[-] maxcount must bigger than 0"
	print_error()

def getrandseed():
	random.seed(time.time())
	random.seed(os.urandom(random.randrange(2**1,2**15)).encode('hex'))

def extractfeat(i, o):
	os.system('python '+py_extractfeat+' -f '+typeFeats+' -i '+i+' -o '+o)

def initdir():
	if os.path.exists(dirTrain) or os.path.exists(dirTest):
		shutil.rmtree(dirTrain)
		shutil.rmtree(dirTest)
	os.mkdir(dirTrain)
	os.mkdir(dirTest)

dirTmp = dirFeat + 'tmp_test/'
dirTrain = dirFeat+'tmp_train_'+str(n)+'/'
dirTest = dirFeat+'tmp_test_'+str(n)+'/'
tagmax = str(global_maxcount) if global_maxcount else "unlimited"

if not os.path.exists(dirTmp):
	os.mkdir(dirTmp)
	extractfeat(dirMan, dirTmp)
	extractfeat(dirWoman, dirTmp)
	extractfeat(dirChild, dirTmp)

listman = []
listwoman = []
listchild = []

listftmp = os.listdir(dirTmp)
for imgname in listftmp:
	imgpath = dirTmp+imgname
	if imgname[0]=='0':
		listchild.append(imgpath)
	if imgname[0]=='1':
		listman.append(imgpath)
	if imgname[0]=='2':
		listwoman.append(imgpath)

while True:
	try:
		getrandseed()
		initdir()
		random.shuffle(listman)
		random.shuffle(listwoman)
		random.shuffle(listchild)
		klistman = random.sample(listman, n)
		klistwoman = random.sample(listwoman, n)
		klistchild = random.sample(listchild, n)
		nklistman = [imgman for imgman in listman if imgman not in klistman]
		nklistwoman = [imgwoman for imgwoman in listwoman if imgwoman not in klistwoman]
		nklistchild = [imgchild for imgchild in listchild if imgchild not in klistchild]
		kimglist = klistman+klistwoman+klistchild
		nkimglist = nklistman+nklistwoman+nklistchild
		for imgpath in kimglist:
			imgname = imgpath.split('/')[-1]
			shutil.copy(imgpath, dirTrain+imgname)
		for imgpath in nkimglist:
			imgname = imgpath.split('/')[-1]
			shutil.copy(imgpath, dirTest+imgname)
		print('[ ] Classifying(n = '+str(n)+')')
		cmdline = shlex.split('python '+py_classifier+' -f '+typeFeats+' -l '+dirTrain+' -t '+dirTest)
		classify = subprocess.Popen(cmdline, stdout=subprocess.PIPE)
		result = re.search('Accuracy: ([0-9\.]+)', classify.communicate()[0])
		if result:
			result = float(result.group(1))
			if not global_average:
				global_average = result
			if result>global_accurate:
				global_accurate = result
				global_kimglist = kimglist
				global_nkimglist = nkimglist
			global_average = (global_average+result)/2
			print('[ ] result  : '+str(result))
			print('[ ] best    : '+str(global_accurate))
			print('[ ] average : '+str(global_average))
		global_counter += 1
		print('[+] ('+str(global_counter)+' / '+tagmax+') case tested')
		print
		if global_maxcount and global_maxcount==global_counter:
			break
	except KeyboardInterrupt:
		print('[-] ^C received, testing ended')
		break

if modelOut:
	print('[!] Saving best Result')
	initdir()
	for imgpath in global_kimglist:
		imgname = imgpath.split('/')[-1]
		shutil.copy(imgpath, dirTrain+imgname)
	for imgpath in global_nkimglist:
		imgname = imgpath.split('/')[-1]
		shutil.copy(imgpath, dirTest+imgname)
	os.system('python '+py_classifier+' -f '+typeFeats+' -M '+modelOut+' -l '+dirTrain+' -t '+dirTest)
