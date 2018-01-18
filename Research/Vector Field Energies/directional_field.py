import math
import numpy as np
import matplotlib.pyplot as plt

# Goal:
# 1. Given a set of vectors or the vector field that it is a subset of, calculate the potential energy associated with the field
#    if the potential energy of a pair of vectors is proportional to the angle between the vectors (and the distance)

# Details:
# Specifically, we transform a vector field f:R^2 --> R^2 into a direction field:
#   g(x,y) = atan2(fy(x,y), fx(x,y))
# We then integrate the square (dot product) of the gradient of g over whatever domain we're interested in.
# This is known as the Dirichlet Energy. It is known that when the Dirichlet energy is minimized, the
# laplacian usually equal to zero (discounting edge cases, or perhaps it's true absolutely, I need to take a pde course).
# Based on the property of a harmonic funcion that f(x, y) = (1/(2 pi r))int_0^2pi f(x + r cos t, y + r sin t) dt, it
# can be deduced that the system is in a steady state.
class Node:
    def __init__(self, pos, value, neighbors):
        self.neighbors = neighbors
        self.value = value
        self.pos = pos

class Field:
    def __init__(self, func, x, y):
        pts = []
        nodes = []
        # sample the domain x in [-10, 10], y in [-10, 10]
        (xStart, xEnd, xSteps) = x
        (yStart, yEnd, ySteps) = y
        xStep = (xEnd - xStart) / xSteps
        yStep = (yEnd - yStart) / ySteps

        i = xStart
        while i <= xEnd + xStep / 2:
            j = yStart
            while j <= yEnd + yStep / 2:
                pts.append([i,j])
                nodes.append(Node([i,j], None, {}))
                j += yStep
            i += xStep

        pts = np.array(pts)

        # link our graph
        count = 0
        (xWidth, yWidth) = (math.floor((xEnd - xStart) / xStep) + 1, math.floor((yEnd - yStart) / yStep) + 1)
        i = xStart
        while i <= xEnd + xStep / 2:
            j = yStart
            while j <= yEnd + yStep / 2:
                if i > xStart + xStep / 2:
                    # link with left node
                    nodes[count].neighbors['left'] = nodes[count - yWidth]
                if i < xEnd - xStep / 2:
                    # link with right node
                    nodes[count].neighbors['right'] = nodes[count + yWidth]
                if j > yStart + yStep / 2:
                    # link with top node
                    nodes[count].neighbors['top'] = nodes[count - 1]
                if j < yEnd - yStep / 2:
                    # link with bottom node
                    nodes[count].neighbors['bottom'] = nodes[count + 1]
                count += 1
                j += yStep
            i += xStep

            self.domainPts = pts
            self.nodes = nodes
            self.func = func
            self.x = (xStart, xEnd, xStep)
            self.y = (yStart, yEnd, yStep)

    def potentialEnergy(self):
        angleFunc = lambda x: math.atan2(x[1], x[0])

        vField = Field.fMap(self.func, self.domainPts, 2)

        angleField = Field.fMap(angleFunc, vField, 1)

        nodes = self.nodes
        (xStart, xEnd, xStep) = self.x
        (yStart, yEnd, yStep) = self.y

        # add angles to nodes
        for i in range(len(angleField)):
            nodes[i].value = angleField[i]
            nodes[i].vector = vField[i]

        # now calculate the sum of the divergence of angleField across the domain.
        # this is (df/dx)^2 + (df/dx)^2
        divergence = 0
        for node in nodes:
            (dfdx, dfdy) = (0,0)
            (dxCount, dyCount) = (0,0)
            for key in ['left', 'right', 'top', 'bottom']:
                if key in node.neighbors:
                    neighbor = node.neighbors[key]
                    angleDiff = abs(neighbor.value - node.value)
                    realDiff = min(angleDiff, 2*math.pi - angleDiff)
                    nodeVec3D = np.array([node.vector[0], node.vector[1], 0])
                    neighborVec3D = np.array([neighbor.vector[0], neighbor.vector[1], 0])
                    sign = 1 if np.cross(nodeVec3D, neighborVec3D)[2] > 0 else -1
                    if key == 'left' or key == 'right':
                        dxCount += 1
                        dfdx += sign * realDiff / (neighbor.pos[0] - node.pos[0])
                    else:
                        dyCount += 1
                        dfdy += sign * realDiff / (neighbor.pos[1] - node.pos[1])

            dfdx /= dxCount
            dfdy /= dyCount

            quarterSectionArea = xStep * yStep / 4
            numKeys = 0
            for key in node.neighbors:
                numKeys += 1
            sectionArea = [0, 0, 1, 2, 4][numKeys] * quarterSectionArea

            divergenceAddition = (dfdx**2 + dfdy**2) * sectionArea
            divergence += divergenceAddition

        return divergence

    # visualization is key
    def drawVecField(self):
        # evaluate f at pts
        codomain = Field.fMap(self.func, self.domainPts, 2)

        # isolate the x and y components
        t = self.domainPts.transpose()
        [domainXs, domainYs] = [t[0], t[1]]
        dt = codomain.transpose()
        [codomainXs, codomainYs] = [dt[0], dt[1]]

        # feed into matplotlib
        plt.quiver(domainXs, domainYs, codomainXs, codomainYs)
        plt.show()

    # a convenience mapping function
    @staticmethod
    def fMap(func, pts, codomainDim):
        c = None
        if codomainDim == 1:
            c = np.empty(len(pts))
        else:
            c = np.empty((len(pts), codomainDim))
        for i in range(len(pts)):
            c[i] = func(pts[i])
        return c

# returns a unit vector in the same direction as the vector, unless the magnitude
# of the vector is zero, in which the zero vector is returned
def normalize(pt):
    n = np.linalg.norm(pt)
    if n == 0:
        return pt
    return pt / n


# an example function
f = lambda x: normalize(np.array( [math.cos(x[0]),.3] ))
field = Field(f, (-10, 10, 21), (-10, 10, 21))
field.drawVecField()

rotate = lambda x,theta: (np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) @ x)

values = []
for i in [10*x for x in range(1,60,2)]:
    field1 = Field(f, (-10, 10, i), (-10, 10, i))
    values.append(field1.potentialEnergy())
    print(values[-1])

plt.plot(values)
plt.show()

values = []
for i in [2*math.pi / 10 * x for x in range(10)]:
    field1 = Field(lambda x: rotate(f(x), i), (-10, 10, 200), (-10, 10, 200))
    energy = field1.potentialEnergy()
    values.append(energy)
    print(values[-1])
