'''
convert from dicts to 32 x n numpy arrays
'''

import numpy as np

class GPSData:
    '''
    '''
    __slots__ = ['data', 'dataFile']

    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.data = self.makeSats
        self.load_data()

    @property
    def makeSats(self):
        '''
        Creates an empty dictionary for each GPS satellite.
        '''
        data = dict([])
        allsats = ['G{:02}'.format(num) for num in range(1, 33)]
        for sat in allsats:
            data[sat] = []
        return data

    def load_data(self):
        with open(self.dataFile, 'r') as f:
            f.readline()
            hdr = f.readline()
            # read 2 lines
            for line1 in f:
                line2 = next(f)
                tmp1 = line1.split()
                tmp2 = line2.split()
                nsats = int(tmp1[1])
                if nsats != -1:
                    sats = tmp1[2:]
                else:
                    sats = sats
                    nsats = len(sats)
                # parse the stations and append data
                for nrec in range(nsats):
                    self.data[sats[nrec]].append([float(tmp1[0]), float(tmp2[nrec])])


