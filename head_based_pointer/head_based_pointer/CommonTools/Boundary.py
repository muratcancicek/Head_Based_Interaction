# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

class Boundary(object):

    def __init__(self, minX = None, maxX = None, minY = None, maxY = None, 
                 minZ = None, maxZ = None, *args, **kwargs):
        minX = float('-inf') if minX == None else float(minX)
        maxX = float('inf') if maxX == None else float(maxX)
        self.__xRange = float('inf') if maxX == None else abs(maxX - minX)
        if minX > maxX: t = minX; minX = maxX; maxX = minX
        minY = float('-inf') if minY == None else float(minY)
        maxY = float('inf') if maxY == None else float(maxY)
        self.__yRange = float('inf') if maxY == None else abs(maxY - minY)
        if minY > maxY: t = minY; minY = maxY; maxY = minY
        minZ = float('-inf') if minZ == None else float(minZ)
        maxZ = float('inf') if maxZ == None else float(maxZ)
        self.__zRange = float('inf') if maxZ == None else abs(maxZ - minZ)
        if minZ > maxZ: t = minZ; minZ = maxZ; maxZ = minZ
        self.__minX, self.__minY, self.__minZ = minX, minY, minZ
        self.__maxX, self.__maxY, self.__maxZ = maxX, maxY, maxZ
        super().__init__()

    def isInRanges(self, x = 0, y = 0, z = 0):
        xIn = self.__minX < x and x < self.__maxX
        yIn = self.__minY < y and y < self.__maxY
        zIn = self.__minZ < z and z < self.__maxZ

        return xIn and yIn and zIn 

    def isIn(self, point):
        z = 0 if len(point) == 2 else point[2]
        return self.isInRanges(point[0], point[1], z)

    def getRanges(self):
        return self.__xRange, self.__yRange, self.__zRange

    def keepInside(self, point):
        if point[0] < self.__minX:
            point[0] = self.__minX + 1
        elif self.__maxX < point[0]:
            point[0] = self.__maxX - 1
        if point[1] < self.__minY:
            point[1] = self.__minY + 1
        elif self.__maxY < point[1]:
            point[1] = self.__maxY - 1
        if len(point) == 2:
            return point
        if point[2] < self.__minZ:
            point[2] = self.__minZ + 1
        elif self.__maxZ < point[2]:
            point[2] = self.__maxZ - 1
        return point

    def flipXYAxes(self):
        minX, minY = self.__minX, self.__minY
        self.__minX, self.__minY = minY, minX

        maxX, maxY = self.__maxX, self.__maxY
        self.__maxX, self.__maxY = maxY, maxX

        t = self.__xRange
        self.__yRange = self.__xRange
        self.__yRange = t
        print('matrix')
        return self

    def getAbsVolume(self, point):
        if len(point) == 2:
            return point - (self.__minX, self.__minY)
        else:
            return point - (self.__minX, self.__minY, self.__minZ)

    def getVolumeRatio(self, point):
        xR = 1 if self.__xRange == float('inf') else self.__xRange
        yR = 1 if self.__yRange == float('inf') else self.__yRange
        if len(point) == 2:
            return point / (xR, yR)
        else:
            zR = 1 if self.__zRange == float('inf') else self.__zRange
            return point / (xR, yR, zR)

    def getVolumeAbsRatio(self, point):
        point = self.getAbsVolume(point)
        return self.getVolumeRatio(point)