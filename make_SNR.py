import numpy as np
import matplotlib.pyplot as plt

elev = np.linspace(0, np.pi / 2, 25)
Ad = 3
Am = 2
lamb = .284
h = 5
phi = 2 * np.pi / lamb * 2 * h * np.sin(elev)
SNR = Ad + Am + 2 * Ad * Am * np.cos(phi)
plt.plot(np.degrees(elev), SNR, '.')
plt.show()

