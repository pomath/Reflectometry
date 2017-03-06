import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# FFZ

def make_sats():
    data = dict([])
    allsats = ['G{:02}'.format(num) for num in range(1, 33)]
    for sat in allsats:
        data[sat] = []
    return data

def load_data( data_file ):
    data = make_sats()
    with open(data_file, 'r') as f:
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
                data[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])
    return data

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
plt.show()
# read in azimuth data



azimuths=load_data('plot_files/rob40110.azi')
elev = load_data('plot_files/rob40110.ele')
snr = load_data('plot_files/rob40110.sn2')


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
plt.show()

key = 'G01'
snrvsele = [(np.sin(np.radians(x[1])), y[1]) for
            x, y in zip(elev[key], snr[key])]
time = [x[0] for x in snr[key] if 3e4 < x[0] < 6e4]
# print(min(time),max(time),(max(time)-min(time))/2)
snrcut = [x for x in snr[key] if
          min(time) <= x[0] <= min(time) + (max(time) - min(time)) / 2]
elecut = [x for x in elev[key] if
          min(time) <= x[0] <= min(time) + (max(time) - min(time)) / 2]
snrg01 = [x[1] for x in snrcut]
eleg01 = [x[1] for x in elecut]

# isolate the lower level of the track
testcut = [[x, y] for x, y in zip(eleg01, snrg01) if float(x) < 30]
el = [x[0] for x in testcut]
sn = [x[1] for x in testcut]

# convert from dbhz to volts
snvolts = np.array([10 ** (x / 20) for x in sn])

# remove the direct power
model = np.polyfit(el, snvolts, 2)
predicted = np.polyval(model, el)
snvolts = snvolts - predicted

# detrend and convert to radians
snvolts = signal.detrend(snvolts)
el = np.sin(np.radians(el))


plt.scatter(el, snvolts, linewidths=.05, s=1)
plt.xlabel('Elevation (sin(e))')
plt.ylabel('Volts')
plt.show()

f = np.linspace(0.01, 1500, 2000)

pgram = signal.lombscargle(np.array(el), np.array(snvolts), f)
plt.figure()
ax = plt.subplot(111)
ax.plot(f, pgram)
plt.xlabel('Frequency 1/sin(e)')
plt.ylabel('Spectral Amp Volts')
plt.show()

phase = (max(*(zip(pgram, f))))
print(phase)
reflection = 4.28018 * np.cos(114.07 * el)
new_sn = snvolts-reflection
plt.figure()
ax = plt.subplot(111)
plt.scatter(el, new_sn, s=1, linewidths=0.05)
plt.show()

pgram = signal.lombscargle(np.array(el), np.array(new_sn), f)
plt.figure()
ax = plt.subplot(111)
ax.plot(f, pgram)
plt.xlabel('Frequency 1/sin(e)')
plt.ylabel('Spectral Amp Volts')
plt.show()
