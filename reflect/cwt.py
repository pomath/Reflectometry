"""
Python sample script for wavelet analysis and the statistical approach
suggested by Torrence and Compo (1998) using the wavelet module. To run
this script successfully, the matplotlib module has to be installed


DISCLAIMER
----------

This module is based on routines provided by C. Torrence and G. P. Compo
available at <http://paos.colorado.edu/research/wavelets/>, on routines 
provided by A. Grinsted, J. Moore and S. Jevrejeva available at
<http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence>, and
on routines provided by A. Brazhe available at
<http://cell.biophys.msu.ru/static/swan/>.

This software is released under a BSD-style open source license. Please read
the license file for furter information. This routine is provided as is without
any express or implied warranties whatsoever.

AUTHORS
-------
Sebastian Krieger, Nabil Freij


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find
from matplotlib.image import NonUniformImage
from snowyWavelet import snowyWavelet
# This script allows different sample data sets to be analysed. Simply comment
# and uncomment the respective fname, title, label, t0, dt and units variables
# to see the different results. t0 is the starting time, dt is the temporal
# sampling step

sample = 'MKEA'
usetex = False
if sample == 'MKEA':
    title = 'Signal'
    fname = 'mkea.dat'
    t0 = 0
    dt = 1
    label = ''
    units = ''
    units2 = ''

'''
if sample == 'NINO3':
    title = 'NINO3 Sea Surface Temperature (seasonal)'
    fname = 'sst_nino3.dat'
    t0 = 1871
    dt = 0.25
    label = 'NINO3 SST'
    if usetex:
        units = r'$^{\circ}\textnormal{C}$'
        units2 = r'$(^{\circ} \textnormal{C})^2$'
    else:
        units = 'degC'
        units2 = 'degC^2'
'''


# set up some data
Ns=1024
#limits of analysis
Nlo=0 
Nhi=Ns
# sinusoids of two periods, 128 and 32.
x=np.arange(0.0,1.0*Ns,1.0)
A=np.sin(2.0*np.pi*x/128.0)
B=np.sin(2.0*np.pi*x/32.0)
A[512:768]+=B[0:256]
var = A

#var = np.loadtxt(fname)

avg1, avg2 = (8, 200)                  # Range of periods to average
slevel = 0.95                        # Significance level

std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (var - var.mean()) / std       # Calculating anomaly and normalizing

N = var.size                         # Number of measurements
time = np.arange(0, N) * dt + t0     # Time array in years

dj = 1/60                            # 30 sub-octaves per octave
s0 = 2 * dt                       # Starting scale, here 2 seconds
J = 7 / dj                        # Seven powers of two with dj sub-octaves
alpha = 0
#alpha, _, _ = wavelet.ar1(var)      # Lag-1 autocorrelation for red noise

mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6

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

# Power rectification as of Liu et al. (2007). TODO: confirm if significance 
# test ratio should be calculated first.
#power /= scales[:, None]

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

# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.
plt.close('all')
plt.ion()
params = {
          'font.size': 13.0,
          'text.usetex': usetex,
          'font.size': 12,
          'axes.grid': True,
         }
plt.rcParams.update(params)          # Plot parameters
figprops = dict(figsize=(11, 8), dpi=72)
fig = plt.figure(**figprops)

# First sub-plot, the original time series anomaly.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(time, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(time, var, 'k', linewidth=1.5)
ax.set_title('a) %s' % (title, ))
if units != '':
  ax.set_ylabel(r'%s [%s]' % (label, units,))
else:
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
bx.fill(np.concatenate([time, time[-1:]+dt, time[-1:]+dt,time[:1]-dt, 
        time[:1]-dt]), (np.concatenate([np.log2(coi),[1e-9], 
        np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]])), 'k', alpha=0.3,
         hatch='x')
bx.set_title('b) %s Wavelet Power Spectrum (%s)' % (label, mother.name))
bx.set_ylabel('Period (seconds)')
#
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), 
                        np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)
#bx.invert_yaxis()
np.save('power.npy', power)
np.save('periods.npy', period)
# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--')
cx.plot(std2*fft_theor, np.log2(period), '--', color='#cccccc')
cx.plot(std2*fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(std2*glbl_power, np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
if units2 != '':
  cx.set_xlabel(r'Power [%s]' % (units2, ))
else:
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
if units != '':
  dx.set_ylabel(r'Average variance [%s]' % (units, ))
else:
  dx.set_ylabel(r'Average variance')
ax.set_xlim([time.min(), time.max()])

fig.savefig('sample.png', format='png', dpi=71)
