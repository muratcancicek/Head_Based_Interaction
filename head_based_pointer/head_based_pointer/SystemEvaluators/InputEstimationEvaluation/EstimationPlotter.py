# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from paths import Experiments_Folder
from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np
import os

def readExperimentData(expName, expFolder = Experiments_Folder):
    source_Folder = expFolder + expName + '/'
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

def readFilteredExperimentData(key, expName, expFolder = Experiments_Folder):
    data = readExperimentData(expName, expFolder)
    return {n: m for n, m in data.items() if key in n or 'WhiteDot' in n}

def readFaceBoxEstimatorsDataFromExperiment(expName, expFolder = Experiments_Folder):
    return readFilteredExperimentData('FaceDetector', expName, expFolder)

def readLandmarkEstimatorsDataFromExperiment(expName, expFolder = Experiments_Folder):
    return readFilteredExperimentData('LandmarkDetector', expName, expFolder)

def readHeadPoseEstimatorsDataFromExperiment(expName, expFolder = Experiments_Folder):
    return readFilteredExperimentData('HeadPoseEstimator', expName, expFolder)

def readEstimatorsDataFromExperiment(expName, expFolder = Experiments_Folder):
    return {'FaceBoxDetectors': readFaceBoxEstimatorsDataFromExperiment(expName, expFolder), 
              'LandmarkDetectors': readLandmarkEstimatorsDataFromExperiment(expName, expFolder),
              'HeadPoseEstimators': readHeadPoseEstimatorsDataFromExperiment(expName, expFolder)}

def getOrderedListOfEstimatorData(outputs):
    items = []
    whiteDot = None
    for n, m in outputs.items():
        if 'WhiteDot' in n:
            whiteDot = (n, m)
        else:
            items.append((n, m))
    if not whiteDot is None: 
        items.append(whiteDot)
    return items

def drawPlots(outputs, num_outputs = 3, show = True): 
    angles = ['X', 'Y', 'Z']
    colors = ['#FFAA00', '#00AA00', '#0000AA', '#AA0000'] 
    f, rows = plt.subplots(num_outputs, 1, sharex=True, figsize=(24, 15))
    #f.suptitle(title)
    items = getOrderedListOfEstimatorData(outputs)
    print(len(items))
    for i in range(num_outputs):
        cell = rows
        lns = []
        if num_outputs > 1: cell = rows[i]
        cell.set_prop_cycle(cycler('color', ['orange', 'blue', 'purple', 'red', 'black']))
        for j, (n, m) in enumerate(items):
            m = m[:, i].reshape((m.shape[0],))
            if 'HeadPoseEstimator' in n:
                m = np.interp(m, (m.min(), m.max()), (-1, +1))
            if 'WhiteDot' in n:
                if i == 2: continue
                cell = cell.twinx()
                l1 = cell.plot(m, 'g--', label=n, linewidth=4.)
            else:
                #print('matrix')
                #print(n, i, m.max(), m.mean())
                l1 = cell.plot(m, label=n)
            lns = l1 + lns
        labs = [l.get_label() for l in lns]
        cell.legend(lns, labs, loc='best')
        cell.set_ylabel('%s ' % (angles[i]))
    if show:
       plt.show()
    f.subplots_adjust(top=0.93, hspace=0, wspace=0)
    return f

def plotEstimatorDataFromExperiment(estimatorType, data, expName, expFolder = Experiments_Folder, show = True, save = True):
    num_outputs = 2
    if 'HeadPoseEstimator' in estimatorType:
        num_outputs = 3
    f = drawPlots(data, num_outputs, show)
    if save:
        fName = Experiments_Folder + expName + '/' + expName + '_' + estimatorType + '.png'
        f.savefig(fName, bbox_inches='tight')

def plotExperiment(expName, expFolder = Experiments_Folder, show = True, save = True):
    estimatorsData = readEstimatorsDataFromExperiment(expName, expFolder)
    #print(estimatorsData)
    for n, d in estimatorsData.items():
        plotEstimatorDataFromExperiment(n, d, expName, expFolder, show, save)

def plot(): 
    #data = readExperimentData()
    expName = 'Exp001'
    #plotExperiment(expName, show = False, save = True)
    data = readHeadPoseEstimatorsDataFromExperiment(expName)
    plotEstimatorDataFromExperiment('HeadPoseEstimators', data, expName, show = False, save = True)

if __name__ == '__main__':
    main()

