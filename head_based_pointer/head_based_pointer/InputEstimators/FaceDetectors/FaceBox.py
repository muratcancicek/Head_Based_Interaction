               
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
        self._tl_corner = (left, top)
        self._tr_corner = (right, top)
        self._bl_corner = (left, bottom)
        self._br_corner = (right, bottom)
        self._width = abs(right - left)
        self._height = abs(bottom - top)
        self.location = (left + self._width/2, top + self._height/2)
        super().__init__(*args, **kwargs)
    
    def getProjectionPoints(self):
        corners = [self._tl_corner, self._tr_corner, self._br_corner, self._bl_corner]
        return [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]

    def isSquare(self):
        return self._width == self._height
        
    def __squareFaceBox(self, f_height, f_width):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        diff = self._width - self._height
        if diff == 1:
            if self._width > self._height:
                if top > 0:
                    return FaceBox(left, top - 1, right, bottom)
                else:
                    return FaceBox(left, top, right, bottom + 1)
            else:
                if left > 0:
                    return FaceBox(left - 1, top, right, bottom)
                else:
                    return FaceBox(left, top, right + 1, bottom)

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
                #print(left, top, right, bottom, self._width, self._height, diff, halfDiff)
                #print(left - halfDiff, top, right + halfDiff, bottom)
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