cimport cython
import numpy as np
cimport numpy as np

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_BOOL = np.bool
ctypedef np.uint8_t DTYPE_BOOL_t

def pnpoly(vert, test):
    vert = np.asarray(vert, dtype=np.float64).atleast_2d()
    test = np.asarray(test, dtype=np.float64).atleast_2d()
    
    return _pnpoly(vert, test, 0, 1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _pnpoly(np.ndarray[DTYPE_FLOAT64_t, ndim=2] vert, np.ndarray[DTYPE_FLOAT64_t, ndim=2] test, int ix, int iy):
    """
    Determine whether m test points are within a polygon defined by a set of 
    n vertices.

    Adapted from the code pnpoly.c by W. Randolph Franklin
    http://www.ecse.rpi.edu/~wrf/Research/Short_Notes/pnpoly.html

    """
 
    cdef int i
    cdef int j
    cdef int k
    cdef int m
    cdef int n
    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] result
   
    n = vert.shape[0]
    m = test.shape[0]

    result = np.zeros(m,DTYPE_BOOL)

    for i in range(n):
        j = (i+n-1) % n
        for k in range(m):
           if(((vert[i,iy] > test[k,iy]) != (vert[j,iy] > test[k,iy])) and \
               (test[k,ix] < (vert[j,ix]-vert[i,ix]) * (test[k,iy]-vert[i,iy])/ \
               (vert[j,iy]-vert[i,iy]) + vert[i,ix]) ):
               result[k] = not result[k]

    return result

def pnpoly3d(np.ndarray[DTYPE_FLOAT64_t, ndim=2] vert, np.ndarray[DTYPE_FLOAT64_t, ndim=2] test):
    """
    Extension of method to 3D.

    As suggested by Franklin, simply project points onto a plane by
    discarding the coordinate with minimal range

    vert and test must be 2d numpy arrays of dtype float64

    Requires (but does not check) that  all vertices and test points 
    are coplanar
    """
    cdef int ix
    cdef int iy
    cdef int discard
    discard = np.argmin(np.max(vert, axis=0)-np.min(vert,axis=0))
    
    #ix = cython.inline("return discard == 0 && 1 || 0")
    #iy = cython.inline("return discard == 2 && 1 || 2")
    if discard == 0:
        ix = 1
        iy = 2
    elif discard == 1:
        ix = 0
        iy = 2
    else:
        ix = 0
        iy = 1

    return _pnpoly(vert, test, ix, iy)
    


