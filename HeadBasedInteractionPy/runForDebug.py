# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
from InteractionDemos.DemoBuilder import run
from InteractionDemos.InputEstimationDemo.InputEstDemoHandler import playInputEst
from InteractionDemos.MappingDemo.MappingDemoHandler import playMapping
from InteractionDemos.InputEstimationDemo.EstimationPlotter import plot
from paths import InputEstimatorsDemo_Folder

def main():
    playMapping()
    #run()
    #playInputEst()
    #plot()

if __name__ == '__main__':
    main()
    print('Done')