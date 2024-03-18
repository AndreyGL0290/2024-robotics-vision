from ntcore import NetworkTableInstance
import numpy as np
import time
import math
import cv2

class NoteConstants:
    # Color range of a note
    color_range = (
        (0, 100, 100),
        (30, 255, 255)
    )
    # Minimum area difference between enclosed circle around the contour and the contour's area
    minAreaDifference = 100
    # Amount of colors that initial frame will be segmented into
    kColors = 5

class NetworkCommunication:
    def __init__(self):
        """
        NetworkCommunications is a class allowing the communication of robot data between the roborio and coproccesser
        via networktables
        """
        self.ntinst = NetworkTableInstance.getDefault()
        self.ntinst.startClient4("wpilibpi")
        self.ntinst.setServerTeam(8775)
        self.ntinst.startDSClient()
        self.rotations_table = self.ntinst.getTable("PIDRotations")

    def send_rotation(self, PIDValue: float):
        self.rotations_table.putNumber("PIDRotation", PIDValue)
        print(PIDValue)

class Note():
    def __init__(self, contour=None, index=None, dA: float = 1000):
        self.cnt = contour
        self.cnt_id = index # Delete when not drawing contour on the original frame
        self.dA = dA
        self.x = None
        self.y = None

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
        frame_mask = self.preprocessed

        contours, _ = cv2.findContours(frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        note = Note()
        PIDValue = None # In case there are no notes in the frame
        maxArea = 0
        for index in range(len(contours)):
            cnt = contours[index]
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            
            # Change the way of circle detection
            _, circle_radius = cv2.minEnclosingCircle(cnt)
            circle_area = circle_radius ** 2 * math.pi
            if area-circle_area < NoteConstants.minAreaDifference and area>maxArea:
                maxArea = area
                note = Note(contour=cnt, index=index, dA=area-circle_area)
                
        frame = self.getFrame()
        if note.inFrame and note.center and self.height:
            PIDValue = (note.center[0] - self.intake_axis) / self.intake_axis

        return (frame, PIDValue)

    @property
    def preprocessed(self):
        frame = self.getFrame()
        # reshaped = frame.reshape((-1,3))
        # reshaped = np.float32(reshaped)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # _ ,label,center=cv2.kmeans(reshaped,8,None,criteria,NoteConstants.kColors,cv2.KMEANS_PP_CENTERS)
        
        # center = np.uint8(center)
        # res = center[label.flatten()]

        # frame = res.reshape((frame.shape))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_mask = cv2.inRange(frame_hsv, NoteConstants.color_range[0], NoteConstants.color_range[1])
        
        return frame_mask

class PipelineRunner:
    def __init__(
            self,
            communications: NetworkCommunication = NetworkCommunication(),
            camera: Camera = None
    ):
        self.communications = communications
        if not camera: self.camera = Camera(cv2.VideoCapture(0))
        else: self.camera = camera

    def run(self, num_of_cycles: int = -1):
        cycle_count = 0
        while cycle_count != num_of_cycles:
            cycle_count += 1

            timestamp = time.time()

            # Processing frame
            _, PIDValue = self.camera.getNote()

            if PIDValue:
                self.communications.send_rotation(PIDValue)

            fps = 1 / ((time.time() - timestamp) or 1e-9)  # prevent divide-by-zero
            print(f"Completed at {time.time() - timestamp}")
            print(f"Cycle {cycle_count} was successful.\nFPS: {round(fps, 3)}")

if __name__ == '__main__':
    PipelineRunner().run()