import cv2
import numpy as np
from constants import *
import math

MatLike = np.ndarray[np.uint8]

class Note():
    def __init__(self, contour: MatLike|None=None, index: int|None = None, dA: float = 1000):
        self.cnt = contour
        self.cnt_id = index # Delete when not drawing contour on the original frame
        self.dA = dA

    @property
    def center(self):
        moments = cv2.moments(self.cnt)
        noteX = int(moments["m10"] / moments["m00"])
        noteY = int(moments["m01"] / moments["m00"])
        return (noteX, noteY)
    
    @property
    def inFrame(self):
        return cv2.moments(self.cnt)["m00"] != 0

class Camera():
    def __init__(self, camera) -> None:
        self.camera: cv2.VideoCapture = camera
        self.height, self.width, _ = self.getFrame().shape
        self.intake_axis = self.width//2

    def getFrame(self):
        return self.camera.read()[1]
        
    def getNote(self):
        frame = self.preprocessed
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_mask = cv2.inRange(frame_hsv, NoteConstants.color_range[0], NoteConstants.color_range[1])

        contours, _ = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        note = Note()

        for index in range(len(contours)):
            cnt = contours[index]
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            
            # Change the way of circle detection
            _, circle_radius = cv2.minEnclosingCircle(cnt)
            circle_area = circle_radius ** 2 * math.pi
            if area-circle_area < note.dA:
                note = Note(contour=cnt, index=index, dA=area-circle_area)
        
        if note.inFrame:
            cv2.drawContours(frame, contours, note.cnt_id, (255,0,0), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, note.center, (0,255,0), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
            cv2.line(frame, note.center, (self.intake_axis, self.height-1), (255,0,0), 2, cv2.LINE_AA)
            cv2.line(frame, note.center, (self.intake_axis, note.center[1]), (255,0,0), 2, cv2.LINE_AA)
            cv2.line(frame, (self.intake_axis, note.center[1]), (self.intake_axis, self.height-1), (255,0,0), 2, cv2.LINE_AA)
            angle = f'{math.asin((note.center[0]-self.intake_axis)/(self.height-note.center[1]-1)):.3f}'
            cv2.putText(frame, angle, (self.intake_axis-100,note.center[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        
        return frame

    def getNoteRotation(self):
        pass

    @property
    def preprocessed(self):
        frame = self.getFrame()
        reshaped = frame.reshape((-1,3))
        reshaped = np.float32(reshaped)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _ ,label,center=cv2.kmeans(reshaped,8,None,criteria,10,cv2.KMEANS_PP_CENTERS)
        
        center = np.uint8(center)
        res = center[label.flatten()]
        preprocessed = res.reshape((frame.shape))
        return preprocessed

    @property
    def color_corrected(self):
        frame = self.getFrame()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) 
        l, a, b = cv2.split(lab) 
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b)) 
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return output