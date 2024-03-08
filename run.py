import time
import cv2

from vision_processing import Camera, NetworkCommunication

class PipelineRunner:
    def __init__(
            self,
            communications: NetworkCommunication = NetworkCommunication(),
            camera: Camera = None
    ):
        self.communications = communications
        if not camera: self.camera = cv2.VideoCapture(1)
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
            print(f"Cycle {cycle_count} was successful.\nFPS: {round(fps, 3)}")