import reflect
import matplotlib.pyplot as plt
import numpy as np
import snowy
import sys
from mpl_toolkits.mplot3d import Axes3D
from array import array
from matplotlib.backends.backend_pdf import PdfPages

def reflect_test():
    '''
    An example usage to create the forward model.
    '''
    R = reflect.Reflect('plot_files/ceri0390', 10, 10)       #Load the tracks from ceri0390
    sat = 'G01'                                         #Pick out the track for G01
    time = [x[0] for x in R.SNR[sat]]                   #Separate out the raw SNR and time
    snr = [x[1] for x in R.SNR[sat]]
    plt.scatter(time, snr, s=1, lw=0)                   #Plot the SNR vs time
    plt.xlim((0, 2e4))
    plt.ylim((20, 60))
    plt.title('SNR for G01 at CERI')
    plt.xlabel('Time (s)')
    plt.ylabel('SNR (dBHz)')
    plt.savefig('G01CERI.eps', format='eps', dpi=1000)  #Save the figure
    R.plotTracks()                                      #Create the skytrack for G01

def refData_test():
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    clipRange=(0, 18000)
    sat = 'G01'

    height, tilt = 10, 10
    t = reflect.refData('plot_files/ceri0390', sat, clipRange, height, tilt)
    Mstart = (1, 0)
    t.retrSat(sat)
    t.fitAd()
    t.clip(clipRange)
    t.CWT()
    newinds = (3737, 3739)
    print(t.clipEle[newinds[0]:newinds[1]], t.clipAzi[newinds[0]:newinds[1]])
    '''
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

def fresnel():
     #FFZ
    n=1
    lamb=3e8*1/1.2e9
    npts=100
    ele=np.linspace(10,15,2)
    es=np.radians(ele)
    az=np.radians(90.0)
    h=1.0
    x_old, y_old = 0, 0
    for e in es:
        d=n*lamb/2
        R=(h/np.tan(e))+(d/np.sin(e))/(np.tan(e))
        b=np.sqrt((2*d*h/np.sin(e))+(d/np.sin(e))**2)
        a=b/np.sin(e)
        print(np.degrees(e))
        theta=np.linspace(0,2*np.pi,npts)
        x_prime=a*np.cos(theta)+R
        y_prime=b*np.sin(theta)
        x=np.sin(az)*x_prime - np.cos(az)*y_prime
        y=np.sin(az)*y_prime + np.cos(az)*x_prime
        plt.plot(x, y)
        plt.plot(x+x_old,y+y_old)
        x_old = x
        y_old = y
        plt.xlabel('meters')
        plt.ylabel('meters')
    plt.show()

def daysheight(datafile, sats, clipRange):
    plt.close('all')
    for sat in sats:
        try:
            height, tilt = 10, 10
            t = reflect.refData(datafile, sat, clipRange, height, tilt)
            t.retrSat(sat)
            t.fitAd()
            t.clip(clipRange)
            t.CWT()
            minh, residual = t.gridSearchH(sat)
            colors = minh / np.max(minh)
            colors[colors == 1] = -1
            colors[colors == 0] = -1
            ax = plt.subplot(111, projection='polar')
            polars = [(x, 90-np.degrees(y)) for x, y in
                      zip(t.clipAzi, t.clipEle)]
            plt.scatter(*zip(*polars), linewidths=.05, s=1.5, c=colors, cmap='Greys')
        except:
            pass
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(90.0)
    ax.set_yticks(range(0, 90, 10))
    ax.set_yticklabels(map(str, range(90, 0, -10)))
    plt.title('Skyplot')
    plt.savefig(t.station + 'height.eps', format='eps', dpi=1000)

def CWTPicture(datafile, sats, clipRange):
    plt.close('all')
    with PdfPages('stations.pdf') as pdf:
        for sat in sats:
            try:
                height, tilt = 10, 10
                t = reflect.refData(datafile, sat, clipRange, height, tilt)
                t.retrSat(sat)
                t.fitAd()
                t.clip(clipRange)
                t.CWT()
                f = t.CWTPlot(t.station + sat + '.png', save = False)
                pdf.savefig(f)
                plt.close('all')
            except:
                pass

def sattracks(sats, datafile, clipRange):
    plt.close('all')
    for sat in sats:
        try:
            height, tilt = 10, 10
            t = reflect.refData(datafile, sat, clipRange, height, tilt)
            t.R.plotTracks(sats, savename = t.station + sat + '.eps')
        except:
            pass

if __name__ == '__main__':
    '''
    - Restrict the elevation angle to a range.
    - Invert every data point using a grid search.
    - Each height/tilt combination is then mapped to its Fresnel Zone.
    '''
    #fresnel()
    station = 'batg'
    yearday = '2017_094'
    folder = 'plot_files/'
    datafile = 'plot_files/batg0940'
    clipRange=(0, 3000)
    snowy.Prep('batg', '2017_094', 'plot_files/')

    if len(sys.argv) == 1:
        sats = ['G{:02}'.format(x) for x in range(1,33)]
    else:
        sats = sys.argv[1:]
    #sattracks(sats, datafile, clipRange)
    CWTPicture(datafile, sats, clipRange)
    #daysheight(datafile, sats, clipRange)
    height, tilt = 10, 10
    t = reflect.Reflect(datafile, height, tilt)
    t.omegaFWD()
    #t.plotOmegaFWD('BATG.png')
