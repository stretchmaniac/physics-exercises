# the goal of this file is to see if a sobel sequence projects well onto arbitrary planes,
# that is, remains a low-descrepancy sequence on that plane
import sobol_seq
import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt

planeNormals = [np.array(x) for x in [
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [0,1,1],
    [1,0,1],
    [1,1,1]
]]

def angle(a,b):
    return math.acos(np.dot(a,b)/(la.norm(a) * la.norm(b)))
def normalize(a):
    return np.multiply(1/la.norm(a), a)

def projectMatrix(planeNormal):
    # we assume for simplicity that the plane passes through the origin
    # we find an orthonormal basis for the plane and combine that with the normal to form a new basis,
    # which we will eventually transform the point to

    # find a vector that is not parallel to planeNormal
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    nonParallel = v2 if angle(v1, planeNormal) < angle(v2, planeNormal) else v1

    perp1 = np.cross(planeNormal, nonParallel)
    perp2 = np.cross(planeNormal, perp1)

    perp3 = normalize(planeNormal)
    perp1 = normalize(perp1)
    perp2 = normalize(perp2)

    return la.inv(np.column_stack([perp1, perp2, perp3]))

def planeCoord(pt, projMat):
    transformed = np.dot(projMat, np.column_stack([pt]))
    return [transformed[0][0], transformed[1][0]]

sobelPts = sobol_seq.i4_sobol_generate(3, 1000)
for k in range(7):
    normal = planeNormals[k]
    mat = projectMatrix(normal)
    pts = []
    for pt in sobelPts:
        if (pt[0]-.5)**2 + (pt[1]-.5)**2 + (pt[2]-.5)**2 < .5**2:
            pts.append(planeCoord(pt, mat))
    [xs, ys] = np.array(pts).transpose()
    plt.subplot(3,3,k+1)
    plt.plot(xs, ys, 'ro')
plt.show()
