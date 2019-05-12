import numpy as np               

class FaceBox(object):
    def __init__(self, left, top, right, bottom, *args, **kwargs):
        if left < right:
            self.left = left  
            self.right = right
        else:
            self.left = right
            self.right = left  
        if top < bottom:
            self.top = top
            self.bottom = bottom
        else:
            self.top = bottom
            self.bottom = top
        self._tl_corner = (self.left, self.top)
        self._tr_corner = (self.right, self.top)
        self._bl_corner = (self.left, self.bottom)
        self._br_corner = (self.right, self.bottom)
        self.width = abs(self.right - self.left)
        self.height = abs(self.bottom - self.top)
        self.location = (self.left + self.width/2, self.top + self.height/2)
        super().__init__(*args, **kwargs)
    
    def getProjectionPoints(self):
        corners = np.array([self._tl_corner, self._tr_corner, self._br_corner, self._bl_corner])
        return corners 

    def isSquare(self):
        return self.width == self.height
        
    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    def __squareFaceBox(self, f_height, f_width):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        if left < 0: left = 0
        if right >= f_width: right = f_width-1
        if top < 0: top = 0
        if bottom >= f_height: bottom = f_height-1

        diff = self.width - self.height
        if diff == 1:
            if self.width > self.height:
                if top > 0:
                    return FaceBox(left, top - 1, right, bottom)
                else:
                    return FaceBox(left, 0, right, bottom + 1)
            else:
                if left > 0:
                    return FaceBox(left - 1, top, right, bottom)
                else:
                    return FaceBox(0, top, right + 1, bottom)

        if abs(diff) % 2 == 1:
            diff += 1 if diff > 0 else -1
        halfDiff = int(diff/2)
        if diff > 0:
            diff, halfDiff = abs(diff), abs(halfDiff)
            if top >= halfDiff and bottom < f_height - halfDiff:
                return FaceBox(left, top - halfDiff, right, bottom + halfDiff)
            elif top < halfDiff:
                return FaceBox(left, 0, right, bottom + (diff - top))
            else:
                return FaceBox(left, (top - diff) + (f_height - bottom), right, f_height)
        else:
            diff, halfDiff = abs(diff), abs(halfDiff)
            if left >= halfDiff and right < f_width - halfDiff:
                return FaceBox(left - halfDiff, top, right + halfDiff, bottom)
            elif left < halfDiff:
                return FaceBox(0, top, right + (diff - left), bottom)
            else:
                return FaceBox((left - diff) + (f_width - right), top, f_width, bottom)

    def getSquareFaceBoxOnFrame(self, frame):
        if self.isSquare():
            return self
        else:
            f_height, f_width = frame.shape[:2]
            squaredFaceBox = self.__squareFaceBox(f_height, f_width)
            return squaredFaceBox.getSquareFaceBoxOnFrame(frame)

    def getFaceImageFromFrame(self, frame):
        #print(self.top,self.bottom, self.left, self.right, '  ')
        return frame[self.top:self.bottom, self.left:self.right]

    def getSquaredFaceImageFromFrame(self, frame):
        squaredFaceBox = self.getSquareFaceBoxOnFrame(frame)
        return squaredFaceBox.getFaceImageFromFrame(frame)