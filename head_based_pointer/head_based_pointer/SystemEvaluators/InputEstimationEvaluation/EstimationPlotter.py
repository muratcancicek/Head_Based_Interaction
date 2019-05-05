# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import Experiments_Folder
import os
import numpy as np
from matplotlib import pyplot as plt

def readExperimentData():
    source_Folder =  Experiments_Folder + 'Exp001/'
    dataFiles = {n[:-4]: open(source_Folder+n) for n in os.listdir(source_Folder) if 'txt' in n}
    data = {}
    for n, f in dataFiles.items():
        #if n == 'Exp001_DLIBHeadPoseEstimator': continue
        mat = []
        for line in f:
            line = line[:-1].replace(', ', '|').split('|')
            line = [float(v) for v in line]
            if 'WhiteDot' in n:
                line.append(0.0)
            mat.append(np.array(line, np.float32))
        data[n] = np.array(mat)
    return data

def readFilteredExperimentData(key):
    data = readExperimentData()
    return {n: m for n, m in data.items() if key in n or 'WhiteDot' in n}

def readFaceBoxDetectorsFromExperimentData():
    return readFilteredExperimentData('FaceDetector')

def readLandmarkDetectorsFromExperimentData():
    return readFilteredExperimentData('LandmarkDetector')

def readHeadPoseEstimatorFromExperimentData():
    return readFilteredExperimentData('HeadPoseEstimator')

def drawPlots(outputs, num_outputs = 3): 
    angles = ['X', 'Y', 'Z']
    colors = ['#FFAA00', '#00AA00', '#0000AA', '#AA0000'] 
    red, blue = (1.0, 0.95, 0.95), (0.95, 0.95, 1.0)
    f, rows = plt.subplots(num_outputs, 1, sharex=True, figsize=(24, 5*num_outputs))
    #f.suptitle(title)
    for i in range(num_outputs):
        cell = rows
        if num_outputs > 1: cell = rows[i]
        for j, (n, m) in enumerate(outputs.items()):
            m = m[:, i].reshape((m.shape[0],))
            m = np.interp(m, (m.min(), m.max()), (-1, +1))
            l1 = cell.plot(m, label=n)
            print(n)
            cell.legend(loc='best')
        cell.set_ylabel('%s ' % (angles[i]))
    plt.show()
    f.subplots_adjust(top=0.93, hspace=0, wspace=0)
    return f

def plot():
    #data = readExperimentData()
    #data = readFaceBoxDetectorsFromExperimentData()
    data = readLandmarkDetectorsFromExperimentData()
    data = readHeadPoseEstimatorFromExperimentData()
    drawPlots(data, num_outputs = 3)

if __name__ == '__main__':
    main()

