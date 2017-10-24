import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelet
from pycwt.helpers import find
from functools import reduce
from .reflectclass import Reflect
from scipy import optimize
import warnings

'''
routines:
extract a full satellite track between a set time range
remove the direct signal (5-10 order polynomial)
'''

class refData:
    '''
    Class that does the heavy lifting.  Uses Reflect objects to invert SNR data
    for the height and elevation of a reflector.
    
    Uses the following internal variables:

    :ivar R: Holds the raw data and creates a synthetic data set based on the
             input height and tilt.
    :vartype R: Reflect

    :ivar Mg: Contains the corresponding model parameters found through the
              inversion.
    :vartype Mg: numpy.ndarray

    :ivar residual: Model parameters residuals.
    :vartype residual: numpy.ndarray


    :param sat: Sattelite to isolate.
    :type sat: str

    :param clipRange: Determines the length of the multipath signal to
                      isolate.  Right now it uses the index instead of an
                      elevation or time.
    :type clipRange: tuple

    :param synthH: Height to use in the synthetic model.
    :type synthH: float

    :param synthT: Tilt to use in the synthetic model.
    :type synthT: float

    Example usage::

        from scipy.optimize import minimize
        clipRange=(0, 18000)
        sat = 'G01'
        
        height, tilt = 10, 10
        t = refData('plot_files/mkea0010', sat, clipRange, height, tilt)
        Mstart = (1, 0)
        t.retrSat(sat)
        t.fitAd()
        t.clip(clipRange)
        t.CWT()
        newinds = (3737, 3739)

        t.loadSynthOmega(sat, clipRange, Mstart)
        res = minimize(t.fullInverse, Mstart, method='nelder-mead',
                       options={'xtol': 1, 'disp': True})
        plt.scatter(t.Mg[::2],
                    np.degrees(t.Mg[1::2]),
                    c=list(range(t.residual.size)))
        plt.xlabel('Height')
        plt.ylabel('Tilt')
        plt.title('Inversion Iterations ' + str(res.nit))
        plt.grid()
        plt.savefig('iterations.eps', format='eps', dpi=1000)
        plt.show()
    '''

    __slots__ = ['R', 'SNR', 'T', 'Ad',
                 'Am', 'clipT', 'clipAm',
                 'cw', 'periods', 'ele',
                 'dele', 'clipEle', 'h',
                 'azi', 'clipAzi', 'samprate',
                 'invH', 'invTilt', 'omg0',
                 'Mg', 'residual', 'error',
                 'freqs', 'datafile', 'clipRange',
                 'trimval', 'wavelength', 'sat',
                 'station']

    def __init__(self, datafile, sat='G01', clipRange=(0, 10000), synthH = 1.0,
                 synthT = 0.0):
        self.datafile = datafile
        self.R = Reflect(datafile, synthH, synthT)
        self.samprate = self.R.samprate
        self.Mg = np.array([])
        self.residual = np.array([])
        self.clipRange = clipRange
        self.wavelength = 0.244
        self.sat = sat
        self.station = datafile[-8:]

    def retrSat(self, sat):
        '''
        Loads the data for a single satellite found in a day's worth of data
        for a single station.  Could probably leave this out and just keep all
        the data in the self.R.SNR variables instead of storing it again.

        :param sat: Satellite to isolate.
        :type sat: str
        '''

        temp = self.R.SNR[sat]
        tmp2 = self.R.elevation[sat]
        tmp3 = self.R.azimuth[sat]
        self.SNR = np.zeros((len(temp), ))
        self.T = np.zeros((len(temp), ))
        self.T = np.array([x[0] for x in temp])
        self.SNR = np.array([x[1] for x in temp])
        self.ele = np.array([np.radians(x[1]) for x in tmp2])
        self.azi = np.array([np.radians(x[1]) for x in tmp3])
        # Convert to Volts????
        self.SNR = self.SNR

    def plotSat(self):
        '''
        Simply plots the SNR after the direct signal has been removed.

        * retrSat(sat)
        * clip(clipRange)
        '''
        plt.plot(np.degrees(self.clipEle), self.clipAm)
        plt.grid()
        plt.show()

    def fitAd(self):
        '''
        Simple routine to remove the direct signal.
        '''
        self.Ad = np.poly1d(np.polyfit(self.T, self.SNR, 10))
        self.Am = self.SNR - self.Ad(self.T)

    def clip(self, clipRange:tuple):
        '''
        Trims data to clipRange, uses indices instead of data values.

        * retrSat(sat)
        * fitAd()
        '''
        self.clipT = self.T - self.T[0]
        start = (np.abs(self.clipT - clipRange[0])).argmin()
        end = (np.abs(self.clipT - clipRange[1])).argmax()
        self.clipT = self.clipT[start:end]
        self.clipAm = self.Am[start:end]
        self.clipEle = self.ele[start:end]
        self.clipAzi = self.azi[start:end]

    def CWT(self):
        '''
        Finds the maximum period at each time step.
        Todo:

        * Need to clean the output more
            * remove 0's
            * only take data in the 95%

        * retrSat(sat)
        * fitAd()
        * clip(clipRange)
        '''
        t0 = 0
        dt = self.samprate
        dj = 1/60
        s0 = 2 * dt
        J = 7 / dj
        var = self.clipAm
        avg1, avg2 = (8, 200)                # Range of periods to average
        slevel = 0.95                        # Significance level
        std = var.std()                      # Standard deviation
        std2 = std ** 2                      # Variance
        var = (var - var.mean()) / std       # Calculating anomaly and normalizing
        N = var.size                         # Number of measurements
        time = np.arange(0, N) * dt + t0     # Time array
        alpha = 0 # White noise
        #alpha, _, _ = wavelet.ar1(var) # Red noise
        mother = wavelet.Morlet(6)
        cwtResults = wavelet.cwt(var, dt, dj, s0, J, mother)
        wave, scales, freqs, coi, fft, fftfreqs = cwtResults
        iwave = wavelet.icwt(wave, scales, dt, dj, mother)
        power = (np.abs(wave)) ** 2
        fft_power = np.abs(fft) ** 2
        period = 1/ freqs
        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                significance_level=slevel, wavelet=mother)
        sig95 = np.ones([1, N]) * signif[:, None]
        sig95 = power / sig95
        # Calculates the global wavelet spectrum and determines its significance level.
        glbl_power = power.mean(axis=1)
        dof = N - scales                     # Correction for padding at edges
        glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                               significance_level=slevel, dof=dof, wavelet=mother)
        # Scale average between avg1 and avg2 periods and significance level
        sel = find((period >= avg1) & (period < avg2))
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        # As in Torrence and Compo (1998) equation 24
        scale_avg = power / scale_avg
        scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                                    significance_level=slevel, dof=[scales[sel[0]],
                                    scales[sel[-1]]], wavelet=mother)
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        maxperiod = np.array([])
        for ind in range(len(coi)):
            per = _nearest(period, coi[ind])
            topind = np.min(np.nonzero(period == per)[0])
            sig95[topind:, ind] = 0
            power[topind:, ind] = 0
            if (sig95[0, ind] and np.where(sig95[:, ind] > 1)):
                maxperiodind = np.argmax(sig95[:, ind])
                maxperiod = np.append(maxperiod, period[maxperiodind])
            else:
                maxperiod = np.append(maxperiod, 0)
        trimval = maxperiod.size // 10
        maxperiod[:trimval] = -1
        maxperiod[-trimval:] = -1
        self.trimval = trimval
        np.save('maxperiod.npy', maxperiod)
        self.periods = maxperiod
        self.freqs = 1 / maxperiod

    def linearInvert(self):
        '''
        Solves for the height given a period.  Assumes zero tilt.

        * retrSat(sat)
        * clip(clipRange)
        * CWT()
        '''
        dt = np.gradient(self.clipT)
        self.dele = np.gradient(self.clipEle, self.samprate)
        self.h = ((self.freqs * self.wavelength / (4 * np.pi * self.samprate )) /
                  (np.cos(self.clipEle) * self.dele))

    def gridSearch(self, sat = 'G01'):
        '''
        Returns the residual given an initial model.
        
        :param Mstart: Initial model parameters.
        :type Mstart: tuple

        * retrSat(sat)
        * clip(clipRange)
        * CWT()
        '''
        self.dele = np.gradient(self.clipEle, self.samprate)
        minh = np.array([])
        minl = np.array([])
        residual = np.array([])
        def point(Mstart, el, dele, freq):
            h, l = Mstart
            synthomg = ((4 * np.pi / self.wavelength) *
                        h * np.cos(el - l) *
                        dele)
            return np.sum(np.abs(synthomg - freq))
        
        mstart = (slice(-10, 100, 1), slice(0, np.pi/2, 0.25))
        print(len(self.clipEle), len(self.dele), len(self.freqs))
        for el, dele, freq in zip(self.clipEle, self.dele, self.freqs):
                
            params = (el, dele, freq)
            
            x0 = optimize.brute(point, mstart, args=params, full_output=True, finish=None)
            minh = np.append(minh, x0[0][0])
            minl = np.append(minl, x0[0][1])
            residual = np.append(residual, x0[1])
        return minh, minl, residual


    def gridSearchH(self, sat = 'G01'):
        '''
        Returns the residual given an initial model.
        
        :param Mstart: Initial model parameters.
        :type Mstart: tuple

        * retrSat(sat)
        * clip(clipRange)
        * CWT()
        '''
        self.dele = np.gradient(self.clipEle, self.samprate)
        minh = np.array([])
        residual = np.array([])
        def point(Mstart, el, dele, freq):
            h = Mstart
            synthomg = ((4 * np.pi / self.wavelength) *
                        h * np.cos(el) *
                        dele)
            return np.sum(np.abs(synthomg - freq))
        
        mstart = (slice(0, 50, 1),)
        print(len(self.clipEle), len(self.dele), len(self.freqs))
        for el, dele, freq in zip(self.clipEle, self.dele, self.freqs):
            params = (el, np.abs(dele), freq)
            x0 = optimize.brute(point, mstart, args=params, full_output=True, finish=None)
            minh = np.append(minh, x0[0])
            residual = np.append(residual, x0[1])
        return minh, residual

    def trim(self):
        '''
        Trims the edges off of the CWT where the values were set to -1.
        '''
        self.clipT = self.clipT[self.trimval:-self.trimval]
        self.clipAm = self.Am[self.trimval:-self.trimval]
        self.clipEle = self.ele[self.trimval:-self.trimval]
        self.clipAzi = self.azi[self.trimval:-self.trimval]
        self.freqs = self.freqs[self.trimval:-self.trimval]

    def fullInverse(self, Mstart = (10, 10)):
        '''
        Solves for the tilt and the height.
        First needs to gather a lot of w for each elevation angle.
        Then construct the Gg matrix with the starting model
        Then check the residual.
        
        - currently just ignores any pinv with a zero singular value
        - Bring the resolution variable out of the interals
            - Sets how much of the data to invert for.

        * retrSat(sat)
        * clip(clipRange)
        '''

        #self.clipPeriod()
        sat= self.sat
        clipRange = (0, 2)
        newinds = (3737, 3739)
        starttilt = np.radians(Mstart[1])
        self.loadSynthOmega(sat, clipRange, Mstart)
        dt = np.gradient(self.clipT[newinds[0]:newinds[1]])
        self.dele = np.gradient(self.clipEle[newinds[0]:newinds[1]], dt)
        const = 4 * np.pi / 0.244 * self.dele
        dh = np.cos(self.clipEle[newinds[0]:newinds[1]] - starttilt)
        dl =  - Mstart[0] * np.sin(self.clipEle[newinds[0]:newinds[1]] - starttilt)
        resolution = 1
        #print(factors(self.clipEle.size), resolution)
        dl = np.split(dl, resolution)
        dh = np.split(dh, resolution)
        fre = np.split(self.freqs[newinds[0]:newinds[1]], resolution)
        omg0 = np.split(self.omg0[newinds[0]:newinds[1]], resolution)
        err = np.split(np.ones((len(self.freqs[newinds[0]:newinds[1]]),)), resolution)
        const = np.split(const, resolution)
        #err = np.split(self.error, resolution)
        for h, l, f, om, e, k in zip(dh, dl, fre, omg0, err, const):
            G = np.array([ k*h, k*l]).T
            U, S, V = np.linalg.svd(G, full_matrices=False)
            if (not np.any(S)):
                pass
            #Gg = np.dot(V, np.dot(np.diag(1/S), U.T))
            #modelRes = np.dot(V, V.T)
            #dataRes = np.dot(U, U.T)
            Gg = np.linalg.pinv(G)
            mg = np.dot(Gg, (f - om))
            dg = k * mg[0] * np.cos(self.clipEle[newinds[0]:newinds[1]] - mg[1])
            res = np.linalg.norm((dg - f)/e) ** 2
            self.residual = np.append(self.residual, res)
            self.invH = mg[0] + Mstart[0]
            self.invTilt = mg[1] + starttilt
            self.Mg = np.append(self.Mg, [self.invH, self.invTilt])
        print(self.Mg[-2], np.degrees(self.Mg[-1]), self.residual[-1], dg[-1], f[-1])
        return self.residual[-1]

    def plotHeight(self):
        '''
        Plots the height on a track.

        * retrSat(sat)
        * clip(clipRange)
        * CWT()
        * linearInvert()
        '''

        print("Plotting height " + u"\u2713")
        ax = plt.subplot(111, projection='polar')
        polars = [(x, np.degrees(np.pi/2-y)) for x, y in
                  zip(self.clipAzi, self.clipEle)]
        Q = plt.scatter(*zip(*polars), c=self.h, s=.5, cmap='viridis')
        cbar = plt.colorbar(Q)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(90.0)
        ax.set_yticks(range(0, 90, 10))
        ax.set_yticklabels(map(str, range(90, 0, -10)))
        plt.show()
        #plt.title(self.label)
        #plt.savefig(self.label + '.eps', format='eps', dpi=1000)
        #print('Saved: ', self.label)

    def HvsT(self):
        '''
        Just makes a plot of the height versus time.

        * linearInvert()
        '''
        plt.plot(self.h,'.')
        #plt.xlim((300, 9000))
        plt.show()

    def loadSynthetic(self, sat, clipRange):
        '''
        Sets all the internal data variables to the forward model.  Height and
        tilt is set in refData.__init__ and sat and clipRange is set here.
        
        :param sat: Satellite to isolate.
        :type sat: str
        
        :param clipRange: Range of indices to isolate.
        :type clipRange: tuple.
        '''
        self.R.omegaFWD()
        self.T = np.array([x[0] for x in self.R.azimuth[sat]])
        freqs = np.array([x for x in self.R.omega[sat]])
        self.azi = np.array([np.radians(x[1]) for x in self.R.azimuth[sat]])
        self.ele = np.array([np.radians(x[1]) for x in self.R.elevation[sat]])
        self.clipT = self.T - self.T[0]
        self.clipT = self.clipT[self.clipT <= clipRange[1]]
        self.freqs = freqs[:self.clipT.size]
        self.clipEle = self.ele[:self.clipT.size]
        self.clipAzi = self.azi[:self.clipT.size]
        #self.error = np.ones((len(self.periods), )) * 0.5

    def loadSynthOmega(self, datafile, sat, clipRange, Mstart = (1, 0)):
        '''
        Used to find the forward model solution for omega when determining the
        residual of self.fullInverse(Mstart).

        * retrSat(sat)
        * clip(clipRange)
        
        
        :param datafile: Path to obs rinex.
        :type datafile: str
        
        :param sat: Satellite that is being inverted.
        :type sat: str
        
        :param clipRange: Range of data to clip.
        :type clipRange: tuple
        
        :param Mstart: Starting model guess for the height and tilt.
        :type Mstart: tuple
        '''
        synthR = Reflect(datafile, Mstart[0], Mstart[1])
        synthR.omegaFWD()
        synthomg = np.array([x for x in synthR.omega[sat]])
        self.omg0 = synthomg[:self.clipT.size]

    def clipPeriod(self):
        '''
        Used to cut the ends off of the SNR data when running self.CWT() so 
        that the more uncertain edges are left out.
        
        * retrSat(sat)
        * clip(clipRange)
        * CWT()
        * loadSynthOmega(datafile,sat,clipRange,Mstart)
        '''
        trimval = self.periods.size // 10
        self.freqs = _trim(self.freqs, trimval)
        self.clipT = _trim(self.clipT, trimval)
        self.clipEle = _trim(self.clipEle, trimval)
        self.omg0 = _trim(self.omg0, trimval)
    
    def CWTPlot(self, savename = 'sample.png', save = True):
        '''
        Creates a plot of the clipped satellite track.
        
        * retrSat(sat)
        * clip(clipRange)
        * 
        '''
        title = self.station + ' ' + self.sat
        t0 = 0
        dt = self.samprate
        label = ''
        var = self.clipAm

        avg1, avg2 = (32, 2048)             # Range of periods to average
        slevel = 0.95                       # Significance level

        std = var.std()                     # Standard deviation
        std2 = std ** 2                     # Variance
        var = (var - var.mean()) / std      # Calculating anomaly and normalizing

        N = var.size                        # Number of measurements
        time = np.arange(0, N) * dt + t0    # Time array in years

        dj = 1/30                           # 30 sub-octaves per octave
        s0 = 2 * dt                         # Starting scale
        J = 10 / dj                         # 10 powers of two with dj sub-octaves
        alpha = 0

        mother = wavelet.Morlet(6)          # Morlet mother wavelet with m=6

        # The following routines perform the wavelet transform and siginificance
        # analysis for the chosen data set.
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J, 
                                                             mother)
        iwave = wavelet.icwt(wave, scales, dt, dj, mother)

        # Normalized wavelet and Fourier power spectra
        power = (np.abs(wave)) ** 2
        fft_power = np.abs(fft) ** 2
        period = 1/ freqs

        # Significance test. Where ratio power/sig95 > 1, power is significant.
        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                significance_level=slevel, wavelet=mother)
        sig95 = np.ones([1, N]) * signif[:, None]
        sig95 = power / sig95

        # Calculates the global wavelet spectrum and determines its significance level.
        glbl_power = power.mean(axis=1)
        dof = N - scales                     # Correction for padding at edges
        glbl_signif, tmp = wavelet.significance(std2, dt, scales, 1, alpha,
                               significance_level=slevel, dof=dof, wavelet=mother)

        # Scale average between avg1 and avg2 periods and significance level
        sel = find((period >= avg1) & (period < avg2))
        Cdelta = mother.cdelta
        scale_avg = (scales * np.ones((N, 1))).transpose()
        # As in Torrence and Compo (1998) equation 24
        scale_avg = power / scale_avg
        scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
        scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                                    significance_level=slevel, dof=[scales[sel[0]],
                                    scales[sel[-1]]], wavelet=mother)

        plt.close('all')
        #plt.ion()
        params = {
                  'font.size': 13.0,
                  'text.usetex': False,
                  'font.size': 12,
                  'axes.grid': True,
                 }
        plt.rcParams.update(params)          # Plot parameters
        figprops = dict(figsize=(11, 8), dpi=40)
        fig = plt.figure(**figprops)

        # First sub-plot, the original time series anomaly.
        ax = plt.axes([0.1, 0.75, 0.65, 0.2])
        ax2 = ax.twiny()
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(np.degrees(self.clipEle))
        ax2.set_xlabel('elevation')
        ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
        ax.plot(time, var, 'k', linewidth=1.5)
        ax.set_title('a) %s' % (title, ))
        ax.set_ylabel(r'%s' % (label, ))
        extent = [time.min(),time.max(),0,max(period)]

        # Second sub-plot, the normalized wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area. Note that period scale is 
        # logarithmic.
        bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        bx.contourf(time, np.log2(period), np.log2(power), np.log2(levels), 
            extend='both', cmap=plt.cm.summer)
        bx.contour(time, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2, 
                   extent=extent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bx.fill(np.concatenate([time, time[-1:]+dt, time[-1:]+dt,time[:1]-dt, 
                    time[:1]-dt]), (np.concatenate([np.log2(coi),[1e-9], 
                    np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]])), 'k', alpha=0.3,
                     hatch='x')

            bx.plot(time, np.log2(self.periods), 'r.')
        bx.plot(time, [np.log2(2 * self.samprate)] * len(time), 'b--')
        bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
        bx.set_ylabel('Period (log2(seconds))')
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), 
                                np.ceil(np.log2(period.max())))
        bx.set_yticks(np.log2(Yticks))
        bx.set_yticklabels(Yticks)

        # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
        # noise spectra. Note that period scale is logarithmic.
        cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
        cx.plot(glbl_signif, np.log2(period), 'k--')
        cx.plot(std2*fft_theor, np.log2(period), '--', color='#cccccc')
        cx.plot(std2*fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
                linewidth=1.)
        cx.plot(std2*glbl_power, np.log2(period), 'k-', linewidth=1.5)
        cx.set_title('c) Global Wavelet Spectrum')

        cx.set_xlabel(r'Power')
        cx.set_xlim([0, glbl_power.max() + std2])
        cx.set_ylim(np.log2([period.min(), period.max()]))
        cx.set_yticks(np.log2(Yticks))
        cx.set_yticklabels(Yticks)
        plt.setp(cx.get_yticklabels(), visible=False)
        #cx.invert_yaxis()


        # Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
        # avg1 and avg2 parameters
        dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
        dx.plot(time, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) $%d$-$%d$ second scale-averaged power' % (avg1, avg2))
        dx.set_xlabel('Time (seconds)')
        dx.set_ylabel(r'Average variance')
        ax.set_xlim([time.min(), time.max()])
        if save:
            fig.savefig(savename, format='png', dpi=71)
        else:
            return fig


def runSynthetic(sat = 'G01'):
    '''
    Used to test the inversion on synthetic data.

    * Add a return value
    
    :param sat: Satellite to isolate.
    :type sat: str
    '''
    clipRange=(0, 12000)

    height, tilt = 10, 10
    t = refData(sat, clipRange, height, tilt)

    #t.R.plotOmegaFWD()
    t.loadSynthetic(sat, clipRange)
    Mstart = (10, 10)
    t.linearInvert()
    #print(t.h)
    Mstart = (t.h[0], 10)
    t.loadSynthOmega(sat, clipRange, Mstart)
    #t.retrSat(sat)
    #t.fitAd()
    #t.clip(clipRange)
    #np.savetxt('snr.dat', t.clipAm)
    #t.CWT()

    t.fullInverse(Mstart)
    Q = plt.scatter(t.Mg[0::2], t.Mg[1::2], c=t.residual, lw=0)
    plt.scatter(height, tilt, marker='x', c='k')
    plt.colorbar(Q)
    plt.xlim((9, 11))
    plt.ylim((9, 11))
    plt.xlabel('Height m')
    plt.ylabel('Tilt (degrees)')
    plt.show()

def _trim(A, trimval):
    A = np.delete(A, np.s_[:trimval])
    A = np.delete(A, np.s_[-trimval:])
    return A

def _nearest(items, pivot):
    '''
    Gets the closest value to pivot in items.
    '''
    return min(items, key=lambda x: abs(x - pivot))

def _factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))



