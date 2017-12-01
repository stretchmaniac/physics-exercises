# see https://mail.google.com/mail/u/1/#search/doss0032%40umn.edu/15fbb88e703a738a?projector=1 for more info

import numpy.random as numpyRand
import numpy as np
import matplotlib.pyplot as plt
import math
import sobol_seq
import numpy.linalg as la

# the total number of sampled points in each trial
samples = 100000

# keeps track of the moments of interia for each trial
cumulativeInertias = []
# the radius of the sphere (m)
r = 1
# the radius of the cylinder (has (0,0,1) as axis) (m)
cR = .3
# the samples will be uniformly distributed in a cube of radius 2r. We want
# integral 1 dV == volume, hence sum_0^samples (1 * sampleVolume) == (2r)^3 ==> sampleVolume = (2r)^3 / samples
sampleVolume = (2*r)**3 / samples

# the density of the sphere, not including the inner cylindrical section (kg / m^3)
sphereDensity = 1
# the density of the inner cylinder (kg / m^3)
cylinderDensity = 2

# the axis for which to compute the moment of inertia around
axis = ((0,0,0),(1,1,0))

def distToAxis(pos):
    # mag((axis[1] - axis[0]) cross (pos - axis[0])) / mag(axis[1] - axis[0]) gives the distance to the axis
    a = (axis[1][0] - axis[0][0], axis[1][1] - axis[0][1], axis[1][2] - axis[0][2])
    b = (pos[0] - axis[0][0], pos[1] - axis[0][1], pos[2] - axis[0][2])
    crossed = (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
    return math.sqrt((crossed[0]**2+crossed[1]**2+crossed[2]**2) / (a[0]**2 + a[1]**2 + a[2]**2))

def densityFunc(pos):
    # if outside the sphere, throw out the point
    if pos[0]**2 + pos[1]**2 + pos[2]**2 >= r**2:
        return 0
    # if inside the cylinder, return cylinder density
    if pos[0]**2 + pos[1]**2 < cR**2:
        return cylinderDensity
    # the only other option is that we're within the sphere but outside the cylinder
    return sphereDensity

# returns a vector in R^3 in the cube of side length 2r centered at the origin, with uniform probability
def rand():
    return (2*r*(numpyRand.rand() - .5), 2*r*(numpyRand.rand() - .5), 2*r*(numpyRand.rand() - .5))

# essentially counts upward in a base that's not 10
def nextbinVec(binVec, dim):
    # start with the third component
    n = 2
    # carry as necessary
    while n >= 0 and binVec[n] + 1 >= dim:
        binVec[n] = 0
        n -= 1
    if n >= 0:
        binVec[n] += 1
    # if n == 0, then we're back at [0,0,0], which is the desired behavior

def angle(a,b):
    return math.acos(np.dot(a,b)/(la.norm(a) * la.norm(b)))
def normalize(a):
    return np.multiply(1/la.norm(a), a)

# given a bin vector, generate a random number in the bin (a small cube somewhere in our original sample space)
def randInbinVec(binVec, dim):
    # transform to the necessary linear value
    [binVecMinX, binVecMinY, binVecMinZ] = [x * (2*r) / dim - r for x in binVec]
    # the width of the mini-cube
    width = (2*r) / dim
    return (binVecMinX + numpyRand.rand() * width, binVecMinY + numpyRand.rand() * width, binVecMinZ + numpyRand.rand() * width)

# we separate the cube we're working in into dim^3 smaller cubes, then sample an appropriate number of points
# from each bin (like a stratified sample)
samplesPerBin = 100
for dim in range(1,20):
    print('binVec '+str(dim)+' started')
    currentbinVec = [0,0,0]
    # start with a moment of inertia of zero (no mass has a moment of inertia of 0)
    cumulativeInertias.append(0)
    binSamples = dim**3 * samplesPerBin
    sampleVolume = (2*r)**3 / binSamples
    for j in range(binSamples):
        # generate a random vector
        pt = randInbinVec(currentbinVec, dim)
        # move to the next bin
        nextbinVec(currentbinVec, dim)
        # add mr^2 to the cumulative moment of inertia. Note that a density value of 0 effectively
        # throws out the point
        cumulativeInertias[-1] += sampleVolume * densityFunc(pt) * distToAxis(pt)**2

# see how much better bins are than no bins
noBinCumulativeInertias = []
for i in range(1,20):
    print('no bin '+str(i)+' started')
    noBinCumulativeInertias.append(0)
    # we want the same number of total plotted points (as before with the bins)
    # however, despite the names, no bins will be made
    binSamples = i**3 * samplesPerBin
    sampleVolume = (2*r)**3 / binSamples
    for j in range(binSamples):
        # see rand defined above
        pt = rand()
        noBinCumulativeInertias[-1] += sampleVolume * densityFunc(pt) * distToAxis(pt)**2

sobolCumulativeInertias = []
# instead of bins, we can try a low discrepancy sequence like the sobol sequence, which I think
# was made for applications like this one
sobolTrials = 20
for j in range(1,sobolTrials):
    print('sobol trial '+str(j))
    # the sobel sequence returns a deterministic list of numbers, thus to test convergence we must
    # vary the number of sampling points
    subSampleSize = j**3 * samplesPerBin
    # recalculated as above
    sampleVolume = (2*r)**3 / subSampleSize

    # generate the first "subSampleSize" numbers of the sobel sequence in R^3
    seq = sobol_seq.i4_sobol_generate(3, subSampleSize)
    # check out sobel_projections.py. In general, the projections onto an arbitrary plane is not garunteed
    # to be good at all... Thus we will just rotate our sobol sequence to line up with the axis (of moment of inertia)

    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    normal = np.array([axis[1][0]-axis[0][0], axis[1][1]-axis[0][1], axis[1][2]-axis[0][2]])
    nonParallel = v2 if angle(v1, normal) < angle(v2, normal) else v1

    perp1 = np.cross(normal, nonParallel)
    perp2 = np.cross(normal, perp1)

    perp3 = normalize(normal)
    perp1 = normalize(perp1)
    perp2 = normalize(perp2)

    for i in range(len(seq)):
        seq[i] = np.dot(np.column_stack([perp1,perp2,perp3]), np.column_stack([seq[i]])).transpose()[0]
    sobolCumulativeInertias.append(0)
    for p in seq:
        sobolCumulativeInertias[-1] += sampleVolume * densityFunc(p) * distToAxis(p)**2

print('Blue: bins, Orange: sobol, Green: no bins')
plt.plot(cumulativeInertias)
plt.plot(sobolCumulativeInertias)
plt.plot(noBinCumulativeInertias)

print('latest results:')
print('no bins: ', noBinCumulativeInertias[-1])
print('bins ', cumulativeInertias[-1])
print('sobol ', sobolCumulativeInertias[-1])

plt.show()
