from forward_model import Reflect
import matplotlib.pyplot as plt
import numpy as np
from Wavelets import Morlet
'''
routines:
extract a full satellite track between a set time range
remove the direct signal (5-10 order polynomial)
'''

class snowyInversion:
    __slots__ = ['R', 'SNR', 'T', 'Ad',
                 'Am', 'clipT', 'clipAm']
    def __init__(self, sat='G26', clipRange=(0, 6000)):
        self.R = Reflect('plot_files/mkea0010', 1.8, 0)
        self.retrSat(sat)
        self.fitAd()
        self.clip(clipRange)

    def retrSat(self, sat:str):
        temp = self.R.SNR[sat]
        self.SNR = np.zeros((len(temp), ))
        self.T = np.zeros((len(temp), ))
        self.T = np.array([x[0] for x in temp])
        self.SNR = np.array([x[1] for x in temp])
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
        self.clipT = self.clipT[self.clipT <= clipRange[1]]
        self.clipAm = self.Am[:self.clipT.size]

    #def waveletAnalysis(self):
        

if __name__ == '__main__':
    import pylab as mpl
    test = snowyInversion()
    A = test.clipAm
    x = test.clipT
    wavelet = Morlet
    notes = 16
    maxscale = 4
    Ns = len(test.clipT)
    Nlo=0 
    Nhi=Ns
    scaling = 'log'
    cw=wavelet(A,maxscale,notes,scaling=scaling)
    scales=cw.getscales()     
    cwt=cw.getdata()
    # power spectrum
    pwr=cw.getpower()
    scalespec=np.sum(pwr,axis=1)/scales # calculate scale spectrum
    # scales
    y=cw.fourierwl*scales
    x=np.arange(Nlo*1.0,Nhi*1.0,1.0)
    
    fig=mpl.figure(1)

    # 2-d coefficient plot
    ax=mpl.axes([0.4,0.1,0.55,0.4])
    mpl.xlabel('Time [s]')
    plotcwt=np.clip(np.fabs(cwt.real), 0., 1000.)
    plotcwt=pwr
    im=mpl.imshow(plotcwt,cmap=mpl.cm.jet,extent=[x[0],x[-1],y[-1],y[0]],aspect='auto')
    #colorbar()
    #ax.set_yscale('log')
    mpl.ylim(y[0],y[-1])
    ax.xaxis.set_ticks(np.arange(Nlo*1.0,(Nhi+1)*1.0,100.0))
    ax.yaxis.set_ticklabels(["",""])
    theposition=mpl.gca().get_position()

    # data plot
    ax2=mpl.axes([0.4,0.54,0.55,0.3])
    mpl.ylabel('Data')
    pos=ax.get_position()
    mpl.plot(x,A,'b-')
    mpl.xlim(Nlo*1.0,Nhi*1.0)
    ax2.xaxis.set_ticklabels(["",""])
    mpl.text(0.5,0.9,"Wavelet example with extra panes",
    fontsize=14,bbox=dict(facecolor='green',alpha=0.2),
    transform = fig.transFigure,horizontalalignment='center')

    # projected power spectrum
    ax3=mpl.axes([0.08,0.1,0.29,0.4])
    mpl.xlabel('Power')
    mpl.ylabel('Period [s]')
    vara=1.0
    #if scaling=="log":
        #mpl.loglog(scalespec/vara+0.01,y,'b-')
    #else:
    mpl.semilogx(scalespec/vara+0.01,y,'b-')
    mpl.ylim(y[0],y[-1])
    mpl.xlim(1000.0,0.01)
    
    mpl.show()
