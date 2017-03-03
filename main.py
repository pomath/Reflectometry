import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# FFZ
n = 1
lamb = 3e8*1/1.2e9
npts = 100
ele = np.linspace(10, 50, 5)
es = np.radians(ele)
az = np.radians(90.0)
h = 1.0
for e in es:
    d = n*lamb/2
    R = (h/np.tan(e))+(d/np.sin(e))/(np.tan(e))
    b = np.sqrt((2*d*h/np.sin(e))+(d/np.sin(e))**2)
    a = b/np.sin(e)

    theta = np.linspace(0, 2*np.pi, npts)
    x_prime = a*np.cos(theta)+R
    y_prime = b*np.sin(theta)
    x = np.sin(az)*x_prime - np.cos(az)*y_prime
    y = np.sin(az)*y_prime + np.cos(az)*x_prime
    plt.plot(x, y)
    plt.xlabel('meters')
    plt.ylabel('meters')

# read in azimuth data
allsats = ['G{:02}'.format(num) for num in range(1, 33)]
azimuths = dict([])
elev = dict([])
snr = dict([])
for sat in allsats:
    azimuths[sat] = []
    elev[sat] = []
    snr[sat] = []
with open('plot_files/rob40110.azi', 'r') as f:
    f.readline()
    hdr = f.readline()
    # read 2 lines
    for line1 in f:
        line2 = next(f)
        tmp1 = line1.split()
        tmp2 = line2.split()
        nsats = int(tmp1[1])
        if nsats != -1:
            sats = tmp1[2:]
        else:
            sats = sats
            nsats = len(sats)
        # parse the stations and append data
        for nrec in range(nsats):
            azimuths[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])

with open('plot_files/rob40110.ele', 'r') as f:
    f.readline()
    hdr = f.readline()
    # read 2 lines
    for line1 in f:
        line2 = next(f)
        tmp1 = line1.split()
        tmp2 = line2.split()
        nsats = int(tmp1[1])
        if nsats != -1:
            sats = tmp1[2:]
        else:
            sats = sats
            nsats = len(sats)
        # parse the stations and append data
        for nrec in range(nsats):
            elev[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])

with open('plot_files/rob40110.sn2', 'r') as f:
    f.readline()
    hdr = f.readline()
    # read 2 lines
    for line1 in f:
        line2 = next(f)
        tmp1 = line1.split()
        tmp2 = line2.split()
        nsats = int(tmp1[1])
        if nsats != -1:
            sats = tmp1[2:]
        else:
            sats = sats
            nsats = len(sats)
        # parse the stations and append data
        for nrec in range(nsats):
            snr[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])

ax = plt.subplot(111, projection='polar')
allsats = ['G01']
for key in allsats:
    if azimuths[key] != []:
        polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                  zip(azimuths[key], elev[key])]
        plt.scatter(*zip(*polars), linewidths=.05, s=.5)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rmax(90.0)
ax.set_yticks(range(0, 90, 10))    # (min int, max int, increment)
ax.set_yticklabels(map(str, range(90, 0, -10)))
plt.title('ROB4-G01 1/11/2017')
