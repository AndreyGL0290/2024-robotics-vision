import cv2
from vision_processing.camera import Camera
from time import perf_counter

if __name__ == '__main__':
    stream = cv2.VideoCapture(1)
    camera = Camera(stream)
    while True:
        time1 = perf_counter()
        frame, PIDValue = camera.getNote()
        time2 = perf_counter()
        fps = 1/((time2-time1)or 1e-9)
        print(f'Frames per Second: {fps}')
        cv2.imshow('note', frame)
        print(f'PIDValue: {PIDValue}')
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    stream.release()
    cv2.destroyAllWindows()