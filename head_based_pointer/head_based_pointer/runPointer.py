# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from SystemEvaluators.InputEstimationEvaluation.InputEstDemoHandler import playInputEst
from SystemEvaluators.MappingEvaluation.MappingDemoHandler import playMapping
from SystemEvaluators.InputEstimationEvaluation.EstimationPlotter import plot
from paths import InputEstimatorsDemo_Folder

def main():
    playMapping()
    #playInputEst()
    #plot()

if __name__ == '__main__':
    main()
