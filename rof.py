from numpy import *

def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
	""" An implementation of the Rudin-Osher_fatemi denosing model
	  using the numerical procedure

	  Input: noisy input image (grayscale), initial guess fo U, weight
	  of the TV-regularizing term, steplength, tolerance for stop criteration.

	  Output: denoised and detextured image, texture residual """

	m,n = im.shape

	U = U_init
	Px = im
	Py = im
	error = 1

	while (error > tolerance):
	  Uold = U

	  # gradient of primal variable
	  GradUx = roll(U,-1,axis=1)-U
	  GradUy = roll(U,-1,axis=0)-U

	  # update the dual variable
	  PxNew = Px + (tau/tv_weight)*GradUx
	  PyNew = Py + (tau/tv_weight)*GradUy
	  NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

	  Px = PxNew/NormNew
	  Py = PyNew/NormNew

	  # update the primal variable
	  RxPx = roll(Px,1,axis=1)
	  RyPy = roll(Py,1,axis=0)

	  DivP = (Px-RxPx)+(Py-RyPy)
	  U = im + tv_weight*DivP

	  # update of error
	  error = linalg.norm(U-Uold)/sqrt(n*m);

	return U,im-U
