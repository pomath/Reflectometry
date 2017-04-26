from Reflect import Reflect
import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelet
from pycwt.helpers import find
from functools import reduce
from scipy.optimize import minimize
'''
routines:
extract a full satellite track between a set time range
remove the direct signal (5-10 order polynomial)
'''

class refData:
    __slots__ = ['R', 'SNR', 'T', 'Ad',
                 'Am', 'clipT', 'clipAm',
                 'cw', 'periods', 'ele',
                 'dele', 'clipEle', 'h',
                 'azi', 'clipAzi', 'samprate',
                 'invH', 'invTilt', 'omg0',
                 'Mg', 'residual', 'error',
                 'freqs']

    def __init__(self, sat='G01', clipRange=(0, 10000), synthH = 1.0, synthT = 0.0):
        self.R = Reflect('../plot_files/ceri0390', synthH, synthT)
        self.samprate = 1
        self.Mg = np.array([])
        self.residual = np.array([])

    def retrSat(self, sat:str):
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
        plt.plot(self.clipT, self.clipAm)
        plt.grid()
        plt.show()

    def fitAd(self):
        self.Ad = np.poly1d(np.polyfit(self.T, self.SNR, 10))
        self.Am = self.SNR - self.Ad(self.T)

    def clip(self, clipRange:tuple):
        self.clipT = self.T - self.T[0]
        self.clipT = self.clipT[clipRange[0]:clipRange[1]]
        self.clipAm = self.Am[:self.clipT.size]
        self.clipEle = self.ele[:self.clipT.size]
        self.clipAzi = self.azi[:self.clipT.size]

    def CWT(self):
        '''
        Finds the maximum period at each time step.
            -Need to clean the output more
                -remove 0's
                -only take data in the 95%
        '''
        t0 = 0
        dt = 1
        dj = 1/60
        s0 = 2 * dt
        J = 7 / dj
        var = self.clipAm
        avg1, avg2 = (8, 200)                  # Range of periods to average
        slevel = 0.95                        # Significance level
        std = var.std()                      # Standard deviation
        std2 = std ** 2                      # Variance
        var = (var - var.mean()) / std       # Calculating anomaly and normalizing
        N = var.size                         # Number of measurements
        time = np.arange(0, N) * self.samprate + t0     # Time array
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
            per = nearest(period, coi[ind])
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
        np.save('maxperiod.npy', maxperiod)
        self.periods = maxperiod
        self.freqs = 1 / maxperiod

    def linearInvert(self):
        '''
        Solves for the height given a period
        '''
        dt = np.gradient(self.clipT)
        self.dele = np.gradient(self.clipEle, dt)
        self.h = (self.freqs * 0.244 / (4 * np.pi * self.samprate )) / (np.cos(self.clipEle) * self.dele)

    def fullInverse(self, Mstart = (10, 10)):
        '''
        Solves for the tilt and the height.
        First needs to gather a lot of w for each elevation angle.
        Then construct the Gg matrix with the starting model
        Then check the residual.
        
        -currently just ignores any pinv with a zero singular value
        '''

        #self.clipPeriod()
        sat='G01'
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
        plt.plot(self.h,'.')
        #plt.xlim((300, 9000))
        plt.show()

    def loadSynthetic(self, sat, clipRange):
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

    def loadSynthOmega(self, sat, clipRange, Mstart = (1, 0)):
        synthR = Reflect('../plot_files/ceri0390', Mstart[0], Mstart[1])
        synthR.omegaFWD()
        synthomg = np.array([x for x in synthR.omega[sat]])
        self.omg0 = synthomg[:self.clipT.size]

    def clipPeriod(self):
        trimval = self.periods.size // 10
        self.freqs = trim(self.freqs, trimval)
        self.clipT = trim(self.clipT, trimval)
        self.clipEle = trim(self.clipEle, trimval)
        self.omg0 = trim(self.omg0, trimval)

    def runSynthetic(self, sat):
        clipRange=(0, 12000)
        sat = 'G01'
        
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

    def pointInverse(self, Mstart):
        '''
        '''

def trim(A, trimval):
    A = np.delete(A, np.s_[:trimval])
    A = np.delete(A, np.s_[-trimval:])
    return A

def nearest(items, pivot):
    '''
    Gets the closest value to pivot in items.
    '''
    return min(items, key=lambda x: abs(x - pivot))

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    clipRange=(0, 18000)
    sat = 'G01'
    
    height, tilt = 10, 10
    t = refData(sat, clipRange, height, tilt)
    #t.loadSynthetic(sat, clipRange)
    Mstart = (1, 0)
    t.retrSat(sat)
    t.fitAd()
    t.clip(clipRange)
    t.CWT()
    #newFreqs = np.loadtxt('freqs.dat')
    newinds = (3737, 3739)
    print(t.clipEle[newinds[0]:newinds[1]], t.clipAzi[newinds[0]:newinds[1]])
    '''
    t.loadSynthOmega(sat, clipRange, Mstart)
    res = minimize(t.fullInverse, Mstart, method='nelder-mead',
                   options={'xtol': 1, 'disp': True})
    plt.scatter(t.Mg[::2], np.degrees(t.Mg[1::2]), c=list(range(t.residual.size)))
    plt.xlabel('Height')
    plt.ylabel('Tilt')
    plt.title('Inversion Iterations ' + str(res.nit))
    plt.grid()
    plt.savefig('iterations.eps', format='eps', dpi=1000)
    plt.show()
    '''
    #np.savetxt('freqs.dat', t.freqs)
    #np.savetxt('snr.dat', t.clipAm)


