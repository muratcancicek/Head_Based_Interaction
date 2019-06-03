from paths import Experiments_Folder
from random import randint
import numpy as np
import cv2

class DotWalker(object):
    
    def __init__(self, size = None, maxFrameCount = 500, backgroundColor = None, dotColor = None, outputFileName = None):
        self.__file = open(Experiments_Folder + outputFileName + '.txt', 'w') if not outputFileName is None else None
        if backgroundColor is None: backgroundColor = (0, 0, 0)
        if dotColor is None: dotColor = (255, 255, 255)
        if size is None: size = (480, 640, 3)
        self.__backgroundColor = backgroundColor
        self.__maxFrameCount = maxFrameCount
        self.__dotColor = dotColor
        self.__pos = np.array((size[1]/2, size[0]/2), np.int32)
        self.__size = size
        self.__frameCount = 0 
        self.__step = 0
        self.__direction = self.getRandomDirection()
        self.__nextCorner = self.getRandomLength()
        super().__init__()

    def getUpdatedFrame(self, frame = None):
        if frame is None: frame = np.zeros(self.__size, dtype = np.uint8)
        self.walk()
        cv2.circle(frame, tuple(self.__pos), 20, self.__dotColor, -1, cv2.LINE_AA)
        return frame

    def isInside(self):
        return self.__pos[0] >= 0 and self.__size[1] >= self.__pos[0] and \
                self.__pos[1] >= 0 and self.__size[0] >= self.__pos[1] 

    def walk(self):
        if self.__frameCount == self.__maxFrameCount:
            raise KeyboardInterrupt
        if self.__step == self.__nextCorner or not self.isInside():
            self.__direction = self.getRandomDirection()
            self.__nextCorner = self.getRandomLength()
            self.__step = 0
        self.__pos += self.__direction
        if not self.__file is None:
            self.__file.write('%d, %d\n' % (self.__pos[0], self.__pos[1]))
        self.__step += 1
        self.__frameCount += 1

    def getRandomDirection(self):
        x = self.getRandomVelocity()
        y = self.getRandomVelocity()
        if x == y and x == 0:
            return self.getRandomDirection()
        if self.__pos[0] < 0:
            x = 1
        elif self.__size[1] < self.__pos[0]:
            x = -1
        if self.__pos[1] < 0:
            y = 1
        if self.__size[0] < self.__pos[1]:
            y = -1

        return np.array((x, y), np.int32)

    def getRandomVelocity(self):
        return randint(-3, 3)

    def getRandomLength(self):
        return randint(50, 250)