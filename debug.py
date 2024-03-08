import cv2
from vision_processing.camera import Camera

if __name__ == '__main__':
    stream = cv2.VideoCapture(1)
    camera = Camera(stream)
    while True:
        frame, PIDValue = camera.getNote()
        cv2.imshow('note', frame)
        print(PIDValue)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    stream.release()
    cv2.destroyAllWindows()