import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import snowy

class Reflect:
    '''
    Contains the forward problem along with some processing for reflectometry.
    '''
    __slots__ = ['azimuth', 'elevation', 'dataFile',
                 'SNR','omega', 'height','maxminOmega',
                 'tilt', 'label']

    def __init__(self, dataFile, height, tilt):
        self.tilt = np.radians(tilt)
        self.dataFile = dataFile
        self.height = height
        self.azimuth = snowy.Data(dataFile + '.azi').data
        self.elevation = snowy.Data(dataFile + '.ele').data
        self.SNR = snowy.Data(dataFile + '.sn1').data
        self.omegaFWD()
        self.label = '../plots/FWD_' + str(self.height) + 'm_' + '{:04.4}deg'.format(np.degrees(self.tilt))

    def plotTracks(self):
        '''
        Plots the satellite tracks recorded at a station.
        '''
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

    def omegaFWD(self):
        '''
        Determines the frequency given a height and sat track.
        '''
        minVal = 1000
        maxVal = 0
        self.omega = dict([])
        for sat in self.elevation:
            dat = self.elevation[sat]
            if dat:
                theta = np.array([y[1] for y in dat],dtype='float64')
                time = np.array([y[0] for y in dat],dtype='float64')
                dt = np.gradient(time)
                dtheta = np.gradient(np.radians(theta), dt)
                self.omega[sat] = (4 * np.pi * self.height)/(0.244) * np.cos(np.radians(theta) - self.tilt) * (dtheta)
                self.omega[sat] = abs(self.omega[sat])
                if min(self.omega[sat]) < minVal:
                    minVal = min(self.omega[sat])
                if max(self.omega[sat]) > maxVal:
                    maxVal = max(self.omega[sat])
        self.maxminOmega = (minVal, maxVal)

    def plotOmegaFWD(self):
        '''
        Plots the frequency on a track.
        '''
        samp = 100
        ax = plt.subplot(111, projection='polar')
        for key in self.elevation:
            if self.azimuth[key]:
                color = [str(item/255.) for item in self.omega[key]]
                polars = [(np.radians(x[1]), 90-y[1]) for x, y in
                          zip(self.azimuth[key], self.elevation[key])]
                Q = plt.scatter(*zip(*polars[::samp]), c=color[::samp], s=.5, cmap='viridis')
        labels = (1 / np.linspace(*(self.maxminOmega), 12)).tolist()
        labels = ['{:4.5}'.format(x) for x in labels]
        cbar = plt.colorbar(Q)
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
        plt.show()
        #plt.savefig(self.label + '.eps', format='eps', dpi=1000)
        #print('Saved: ', self.label)

    def saveSNR(self):
        '''
        Saves synthetic data to be run through the inversion.
        '''
        

    def getFullTrack(self):
        '''
        Returns a full satellite pass.
        '''
        print('hello')

if __name__ == '__main__':
    '''
    An example usage to create the forward model.
    '''
    R = Reflect('../plot_files/ceri0390', 1.8, 0)
    R.plotOmegaFWD()































