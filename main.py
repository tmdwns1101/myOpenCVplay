from trackingObject.TrackingObject import trackingObject
from makeCircle.MakeCircle import  makeCircle
import cv2


cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # CV_CAP_PROP_FRAME_WIDTH
cam.set(4, 720)  # CV_CAP_PROP_FRAME_HEIGHT


def main():
    trackingObject(cam)
    #makeCircle(cam)

if __name__=="__main__":
    main()