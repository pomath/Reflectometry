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

sp3file = 'igs19300.sp3'
SAT = int_sp3(sp3file, 30)

#def azelle(S, P):
P = np.array([-371052.136, -1676729.269, -6122039.258]) #Backer Island
S = SAT['G32']
Xs = S[1,:]
Ys = S[2,:]
Zs = S[3,:]
XR = P[0]
YR = P[1]
ZR = P[2]


