import numpy as np
import matplotlib.pyplot as plt

def make_sats():
    '''
    Creates an empty dictionary for each GPS satellite.
    '''
    data = dict([])
    allsats = ['G{:02}'.format(num) for num in range(1, 33)]
    for sat in allsats:
        data[sat] = []
    return data


def load_data(data_file):
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

def plot_all_tracks(azi, elev):
    ax = plt.subplot(111, projection='polar')
    for key in azi:
        if azi[key] != []:
            polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                      zip(azi[key], elev[key])]
            plt.scatter(*zip(*polars), linewidths=.05, s=.5)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(90.0)
    ax.set_yticks(range(0, 90, 10))    # (min int, max int, increment)
    ax.set_yticklabels(map(str, range(90, 0, -10)))
    plt.title('ROB4 1/11/2017')
    plt.show()

def get_snr_freq(elev,h):
    omega = dict([])
    for sat in elev:
        dat = elev[sat]
        theta = np.array([y[1] for y in dat],dtype='float64')
        time = np.array([y[0] for y in dat],dtype='float64')

        dt = np.diff(time)
        dtheta = np.diff(theta)
        omega[sat] = (4 * np.pi * h)/(0.244) * np.cos(np.radians(theta[1:])) * (dtheta/dt)
        omega[sat] = abs(omega[sat])
    return omega

def plot_omega(omega,azi,elev):
    ax = plt.subplot(111, projection='polar')

    for key in elev:
        if azi[key] != []:
            color = [str(item/255.) for item in omega[key]]
            polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                      zip(azi[key], elev[key])]
            Q = plt.scatter(*zip(*polars[0::15]), c=color[0::15], s=.5, cmap = 'Pastel1_r')


    ax.set_theta_zero_location('N')
    cbar = plt.colorbar(Q,ticks=[])
    cbar.ax.invert_yaxis()
    ax.set_theta_direction(-1)
    ax.set_rmax(90.0)
    ax.set_yticks(range(0, 90, 10))    # (min int, max int, increment)
    ax.set_yticklabels(map(str, range(90, 0, -10)))
    plt.title('ROB4 1/11/2017')
    plt.show()
    #plt.savefig('ROB4-FWD.eps',format='eps', dpi=1000)
azimuths = load_data('plot_files/rob40110.azi')
elev = load_data('plot_files/rob40110.ele')
omega30 = get_snr_freq(elev,30.)
plot_omega(omega30,azimuths,elev)































