import cv2

class PeopleRectDetector:
    '检测行人所在矩形区域'

    def __init__(self):
        self.peoplehog = cv2.HOGDescriptor()
        self.peoplehog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

        self.hitThreshold = 0
        self.winStride = (4,4)
        self.padding = (32,32)
        self.scale = 1.1
        self.finalThreshold = 2

    def setParas(self,hitThreshold = 0,winStride = (4,4),padding = (32,32),scale = 1.1,finalThreshold = 2):
        self.hitThreshold = hitThreshold
        self.winStride = winStride
        self.padding = padding
        self.scale = scale
        self.finalThreshold = finalThreshold

    def detect(self,img):
        (peopleRects,_) = self.peoplehog.detectMultiScale(img,
                                                          self.hitThreshold,
                                                          self.winStride,
                                                          self.padding,
                                                          self.scale,
                                                          self.finalThreshold)
        return peopleRects