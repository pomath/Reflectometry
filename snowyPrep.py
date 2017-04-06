'''
download data
make the plot files
load into numpy arrays and save as binaries
'''

import subprocess


class snowyPrep:
    '''
    Used to go from a zipped rinex file to the plot files from TEQC.
    -need to add more checks to make it dummy proof.
    '''
    __slots__ = ['rawFP', 'zipFP', 'obsFP',
                 'crxFP', 'runTEQC', 'runCRX2RNX',
                 'workPath', 'found']

    def __init__(self, station='min0', date='2008_001',
                 path="plot_files/"):
        self._workingDir()
        self.rawFP = (self.workPath + '/' + path
                      +
                      station + date[-3:] +
                      '0.' + date[2:4])
        self.zipFP = self.rawFP + 'd.Z'
        self.obsFP = self.rawFP + 'o'
        self.crxFP = self.rawFP + 'd'
        #self.preProc()

    def _workingDir(self):
        self.workPath = subprocess.run(["pwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.workPath = self.workPath.stdout[:-1]
        self.workPath = self.workPath.decode('ascii')

    def fileExists(self):
        for path in [self.zipFP, self.obsFP, self.crxFP]:
            self.found = subprocess.run(["ls", path.encode('ascii')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if (self.found.returncode != 0):
                print("Didn't find ", path)
            else:
                print("Found ", path)

    def preProc(self):
        subprocess.run(["gzip", "-df", self.zipFP])
        self.runCRX2RNX = subprocess.run(["./CRX2RNX", self.crxFP], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.runTEQC = subprocess.run(["./teqc", "+qcq", "+plot", self.obsFP], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.failed()

    def failed(self):
        if self.runCRX2RNX.returncode or self.runTEQC.returncode:
           print('Prep failed for ' + self.rawFP)


if __name__ == '__main__':
    test = snowyPrep('mkea', '2005_001')
    test.fileExists()


