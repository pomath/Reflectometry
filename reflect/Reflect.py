import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import snowy


class Reflect:
    '''
    Class containing routines used to set up the inversion.
    
    Uses the following internal variables:
    
    :ivar azimuth: Dictionary containing a list of azimuth
                   and epoch information for each satellite.
    :type azimuth: dict
    :ivar elevation: Dictionary containing a list of elevation
                     and epoch information for each satellite.
    :type elevation: dict
    :ivar dataFile: Path to the folder containing the .azi, .ele
                    and .sn1 files.
    :type dataFile: str
    :ivar SNR: Dictionary containing a list of SNR and epoch
               information for each satellite.
    :type SNR: dict
    :ivar omega: Dictionary containing the calculated values for
                 the multipath frequency.
    :type omega: dict
    :ivar height: Simulated height of a reflector nearby the antenna
                  in meters.
    :type height: float
    :ivar maxminOmega: Contains the maximum and minimum values of omega
                       to use for the plot colorbar.
    :type maxminOmega: tuple
    :ivar tilt: Simulated tilt of reflector nearby the antenna in
                degrees.
    :type tilt: float
    :ivar label: Used as the title for plots and saving plots.
    :type label: str
    '''
    __slots__ = ['azimuth', 'elevation', 'dataFile',
                 'SNR', 'omega', 'height',
                 'maxminOmega', 'tilt', 'label']

    def __init__(self, dataFile, height, tilt):
        self.tilt = np.radians(tilt)
        self.dataFile = dataFile
        self.height = height
        self.azimuth = snowy.Data(dataFile + '.azi').data
        self.elevation = snowy.Data(dataFile + '.ele').data
        self.SNR = snowy.Data(dataFile + '.sn1').data
        self.omegaFWD()
        self.label = ('../plots/FWD_' + str(self.height) + 'm_' +
                      '{:04.4}deg'.format(np.degrees(self.tilt)))

    def plotTracks(self):
        '''
        Plots the satellite tracks recorded at a station.
        '''
        ax = plt.subplot(111, projection='polar')
        for key in ['G01']:
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
        plt.savefig('G01CERITrack.eps', format='eps', dpi=1000)

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
                theta = np.array([y[1] for y in dat], dtype='float64')
                time = np.array([y[0] for y in dat], dtype='float64')
                dt = np.gradient(time)
                dtheta = np.gradient(np.radians(theta), dt)
                self.omega[sat] = ((4 * np.pi * self.height)/(0.244) *
                                   np.cos(np.radians(theta) - self.tilt) *
                                   (dtheta))
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
                Q = plt.scatter(*zip(*polars[::samp]),
                                c=color[::samp],
                                s=2,
                                cmap='viridis',
                                lw=0)
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
        plt.savefig(self.label + '.eps', format='eps', dpi=1000)
        print('Saved: ', self.label)

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
    R = Reflect('../plot_files/ceri0390', 10, 10)
    sat = 'G01'
    time = [x[0] for x in R.SNR[sat]]
    snr = [x[1] for x in R.SNR[sat]]
    plt.scatter(time, snr, s=1, lw=0)
    plt.xlim((0, 2e4))
    plt.ylim((20, 60))
    plt.title('SNR for G01 at CERI')
    plt.xlabel('Time (s)')
    plt.ylabel('SNR (dBHz)')
    plt.savefig('G01CERI.eps', format='eps', dpi=1000)
    R.plotTracks()
