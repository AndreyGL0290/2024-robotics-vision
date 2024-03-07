import cv2
from vision_processing.camera import Camera

if __name__ == '__main__':
    stream = cv2.VideoCapture(1)
    camera = Camera(stream)
    while True:
        cv2.imshow('camera stream', camera.getNote())
        # camera.getNote()
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    stream.release()
    cv2.destroyAllWindows()