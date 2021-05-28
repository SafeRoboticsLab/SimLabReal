import numpy as np

def euler2rot(a):
	a = np.asarray(a).flatten()
	ch = np.cos(a[1])
	sh = np.sin(a[1])
	ca = np.cos(a[0])
	sa = np.sin(a[0])
	cb = np.cos(a[2])
	sb = np.sin(a[2])
	R = np.zeros((3,3))
	R[0,0] = ch * ca
	R[0,1] = sh*sb - ch*sa*cb
	R[0,2] = ch*sa*sb + sh*cb
	R[1,0] = sa
	R[1,1] = ca*cb
	R[1,2] = -ca*sb
	R[2,0] = -sh*ca
	R[2,1] = sh*sa*cb + ch*sb
	R[2,2] = -sh*sa*sb + ch*cb
	return R
