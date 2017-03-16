import numpy as np


def read_sp3(sp3file, sv):
    Ts = np.array([])
    Xs = np.array([])
    Ys = np.array([])
    Zs = np.array([])
    with open(sp3file) as f:
        lastt = []
        for line in f:
            if (line[0] == '*'):
                t = line.split()[1:]
                lastt = [float(x) for x in t]
                lastt = lastt[3] * 3600. + lastt[4] * 60. + lastt[5]
            if (line[0] == 'P'):
                tmp = line.split()
                tmp_sv = int((tmp[0])[2:4])
                if tmp_sv == sv:
                    Ts = np.append(Ts, lastt)
                    Xs = np.append(Xs, float(tmp[1]))
                    Ys = np.append(Ys, float(tmp[2]))
                    Zs = np.append(Zs, float(tmp[3]))
    return Ts, Xs, Ys, Zs


def int_sp3(sp3file, splint):

    R = 6471000

    SAT = dict([])
    nhdr = 2
    splint = 30
    with open(sp3file) as f:
        for dummy_line in range(nhdr):
            f.readline()
        sats1 = (f.readline().split())[2]
        sats2 = (f.readline().split())[1]
        keys = [ sats1[start:start+3] for start in range(0, len(sats1), 3)]
        keys.extend([ sats2[start:start+3] for start in range(0, len(sats2), 3)])

    #print(keys)
    for sv in keys:
        Ts, Xs, Ys, Zs = read_sp3(sp3file, float(sv[1:]))
        Xs = Xs * 1000
        Ys = Ys * 1000
        Zs = Zs * 1000
        Ti = np.linspace(0, 25*3600, 25 * 3600  / splint + 1)
        Xi = np.interp(Ti, Ts, Xs)
        Yi = np.interp(Ti, Ts, Ys)
        Zi = np.interp(Ti, Ts, Zs)
        
        T = Ti / 3600
        SAT[sv] = np.array([])
        SAT[sv] = np.vstack((T, Xi, Yi, Zi))
    return SAT


def xyz2wgs(S):

    a = 6378137.0
    b = 6356752.314
    f = 1.0 / 298.257222101
    eo = 2 * f - f ** 2
    el = (a ** 2 - b ** 2) / b ** 2
    t = S[:,0]
    x = S[:,1]
    y = S[:,2]
    z = S[:,3]
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)
    phi = np.arctan2(z + np.sin(theta)**3 * el * b, p - np.cos(theta)**3 * eo **2 * a)
    lam = np.arctan2(y, x)

    N = a ** 2 / np.sqrt(np.cos(phi)**2 * a**2 + np.sin(phi) ** 2 * b ** 2)
    alt = (p / np.cos(phi)) - N
    R = np.vstack((t, lam * 180 / np.pi, phi * 180 / np.pi, alt)).T
    return R

def azelle(S, P):

    
    Xs = S[1,:]
    Ys = S[2,:]
    Zs = S[3,:]
    XR = P[0]
    YR = P[1]
    ZR = P[2]

    Xs = np.subtract(Xs, XR)
    Ys = np.subtract(Ys, YR)
    Zs = np.subtract(Zs, ZR)

    Xs = np.reshape(Xs, (len(Xs),1))
    Ys = np.reshape(Ys, (len(Ys),1))
    Zs = np.reshape(Zs, (len(Zs),1))



    rang = np.sqrt(Xs ** 2 + Ys ** 2 + Zs ** 2)

    Xu = Xs / rang
    Yu = Ys / rang
    Zu = Zs / rang

    Ru = np.hstack((Xu, Yu, Zu))
    return Ru


# Working Area
P = np.array([-371052.136, -1676729.269, -6122039.258]) #Backer Island
sp3file = 'igs19300.sp3'
SAT = int_sp3(sp3file, 30)
S = SAT['G32']

Ru = azelle(S, P)
O = P
V = Ru
#def xyz2neu(O, V):

X = V[:,0]
Y = V[:,1]
Z = V[:,2]

NEU = np.array([])
R = np.ones_like(V)
XR = R[:,0] * O[0]
YR = R[:,1] * O[1]
ZR = R[:,2] * O[2]

T = np.zeros_like(XR)

Er = np.vstack((T, XR, YR, ZR)).T
E = xyz2wgs(Er)

E[:,1] = E[:,1] * np.pi/180.
E[:,2] = E[:,2] * np.pi/180.

cp = np.cos(E[:,1])
sp = np.sin(E[:,1])
cl = np.cos(E[:,2])
sl = np.sin(E[:,2])























