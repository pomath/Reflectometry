import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

class GPSData:
    '''
    '''
    __slots__ = ['data', 'dataFile']

    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.data = self.makeSats
        self.load_data()

    @property
    def makeSats(self):
        '''
        Creates an empty dictionary for each GPS satellite.
        '''
        data = dict([])
        allsats = ['G{:02}'.format(num) for num in range(1, 33)]
        for sat in allsats:
            data[sat] = []
        return data

    def load_data(self):
        with open(self.dataFile, 'r') as f:
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
                    self.data[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])


class Reflect:
    '''
    '''
    __slots__ = ['azimuth', 'elevation', 'dataFile',
                 'SNR','omega', 'height','maxminOmega',
                 'tilt', 'label']

    def __init__(self, dataFile, height, tilt):
        self.tilt = np.radians(tilt)
        self.dataFile = dataFile
        self.height = height
        self.azimuth = GPSData(dataFile + '.azi').data
        self.elevation = GPSData(dataFile + '.ele').data
        self.SNR = GPSData(dataFile + '.sn1').data
        self.get_omega()
        self.label = 'FWD_' + str(self.height) + 'm_' + '{:4.4}deg'.format(np.degrees(self.tilt))

    def plot_all_tracks(self):
        ax = plt.subplot(111, projection='polar')
        for key in self.azimuth:
            if self.azimuth[key] != []:
                polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                          zip(self.azimuth[key], self.elevation[key])]
                plt.scatter(*zip(*polars), linewidths=.05, s=.5)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(90.0)
        ax.set_yticks(range(0, 90, 10))
        ax.set_yticklabels(map(str, range(90, 0, -10)))
        plt.title(self.dataFile + ' Height: ' + str(self.height) + 'm')
        plt.show()

    def get_omega(self):
        minVal = 1000
        maxVal = 0
        self.omega = dict([])
        for sat in self.elevation:
            dat = self.elevation[sat]
            if dat:
                theta = np.array([y[1] for y in dat],dtype='float64')
                time = np.array([y[0] for y in dat],dtype='float64')
                dt = np.gradient(time)
                dtheta = np.gradient(theta)
                self.omega[sat] = (4 * np.pi * self.height)/(0.244) * np.cos(np.radians(theta) - self.tilt) * (dtheta/dt)
                self.omega[sat] = abs(self.omega[sat])
                if min(self.omega[sat]) < minVal:
                    minVal = min(self.omega[sat])
                if max(self.omega[sat]) > maxVal:
                    maxVal = max(self.omega[sat])
        self.maxminOmega = (minVal, maxVal)

    def plot_omega(self):
        '''
        Plots the frequency on a track.
        '''
        ax = plt.subplot(111, projection='polar')
        for key in self.elevation:
            if self.azimuth[key]:
                color = [str(item/255.) for item in self.omega[key]]
                polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                          zip(self.azimuth[key], self.elevation[key])]
                Q = plt.scatter(*zip(*polars[0::15]), c=color[0::15], s=.5, cmap='Pastel1_r')
        labels = (1 / np.linspace(*(self.maxminOmega), 12)).tolist()
        labels = ['{:4.2}'.format(x) for x in labels]
        cbar = plt.colorbar(Q, spacing='uniform')
        tick_locator = ticker.MaxNLocator(nbins=12)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.invert_yaxis()
        cbar.set_label('Period (s)')
        cbar.ax.set_yticklabels(labels)
        # Add some numbers to the colorbar
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(90.0)
        ax.set_yticks(range(0, 90, 10))
        ax.set_yticklabels(map(str, range(90, 0, -10)))
        plt.title(self.label)
        plt.savefig(self.label.replace('.','_') + '.eps', format='eps', dpi=1000)
        print('Saved: ', self.label)

    def getFullTrack(self):
        '''
        Returns a full satellite pass.
        '''

if __name__ == '__main__':
    '''
    '''
    R = Reflect('plot_files/rob40110', 3, 0)
    R.plot_omega()































