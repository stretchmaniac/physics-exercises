import numpy as np
import matplotlib.pyplot as plt

# the earth is stationary and the moon follows a set trajectory. Thus
# we only need to integrate the position of a single object -- the satellite.

# all units are standard SI units (kg/s/m, etc)

earth = {
    'pos':np.array([0,0,0]),
    'mass':5.9736*10**24,
}

moon = {
    'mass':7.3477*10**22,
    'pos':None,
    'alpha':0,
    'history':[]
}

satellite = {
    # somewhat arbitrary mass
    'mass':100,
    'pos':None,
    'vel':None,
    'history':[]
}

# siderial day
periodS = 86164

# the constant
G = 6.6743 * 10**(-11)

# distance from center of earth
geoStationaryRadius = (G*earth['mass']*periodS**2 / (4*math.pi**2))**(1/3.0)

# set the position of the satellite
radiusFactor = 1
satellite['pos'] = radiusFactor * np.array([geoStationaryRadius, 0, 0])
# perpendicular to the position vector, in the positive y direction
satellite['vel'] = np.array([0, math.sqrt(G*earth['mass'] / (radiusFactor * geoStationaryRadius)), 0])

moonOrbitPeriod = 27.25 * periodS

# distance from center of earth
moonRadius = 384400000

# since this system is relatively consistent in terms of magnitudes of forces, I won't bother
# with a changing dt
n = 1000
dt = periodS / n

time = 0
# first number is the number of days
endTime = 1 * 24 * 60 * 60

def moonPos(t):
    omega = 2*math.pi / moonOrbitPeriod
    return moonRadius * np.array([math.cos(moon['alpha'])*math.cos(omega*t), math.sin(omega*t), math.sin(moon['alpha'])*math.cos(omega*t)])

def cartesianToSpherical(pos):
    r = np.linalg.norm(pos)
    theta = math.atan2(pos[1], pos[0])
    phi = math.acos(pos[2] / r)
    return np.array([r,theta,phi])

moon['pos'] = moonPos(time)

while time < endTime:
    # update position, then acceleration, then velocity
    satellite['pos'] += satellite['vel'] * dt

    # calculate the acceleration due to the moon and earth
    f_g = np.array([0.0,0.0,0.0])
    for el in [earth, moon]:
        f_g += (el['pos'] - satellite['pos']) * G*el['mass']*satellite['mass'] / np.linalg.norm(el['pos'] - satellite['pos'])**2
    acc = f_g / satellite['mass']

    # update velocity
    satellite['vel'] += acc * dt

    # I'm not entirely sure whether to update the history before or after pos/vel is updated.
    # I think, since the object will not move the first iteration, we'll do it after the computation
    p = satellite['pos']
    satellite['history'].append(np.array([p[0], p[1], p[2]]))
    p = moon['pos']
    moon['history'].append(np.array([p[0], p[1], p[2]]))

    # update the moon position
    moon['pos'] = moonPos(time)

    time += dt

for el in [satellite, moon]:
    el['history'] = [x[:2] for x in el['history']]

plt.scatter(*np.array(satellite['history']).transpose())
plt.scatter(*np.array(moon['history']).transpose())
plt.show()
