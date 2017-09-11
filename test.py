import reflect
import matplotlib.pyplot as plt
import numpy as np

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
    
if __name__ == '__main__':
    reflect_test()
    print('reflect test succeeded')
    refData_test()
    print('refData_test suceeded')
