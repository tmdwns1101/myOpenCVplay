import imutils
import cv2
from imutils.video import FPS
import random
import time
from stage.StageMaker import *
def drawCircle(centers):


    for center in centers:
        center[0] = random.randrange(1, 1000)
        center[1] = random.randrange(1, 480)
        #print(center[0], center[1])

    #print("--------")
    return centers


def trackingObject(cam):

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
         "mosse": cv2.TrackerMOSSE_create
    }

    tracker = OPENCV_OBJECT_TRACKERS["kcf"]()

    initBB = None
    fps = None

    changeFlag = False
    clearFlag = False
    newStageFlag = False   #새 스테이지 진입 했는지 확인

    stage = 1 # init stage
    centers, colors, target_score, stage = searchStage(stage)

    for center in centers:
        center[0] = random.randrange(40, 1000)
        center[1] = random.randrange(40, 700)


    score = 0

    while True:
        ret, frame = cam.read()
        if ret:
            frame = imutils.resize(frame, width=1000)
            (H, W) = frame.shape[:2]




            if initBB is not None:

                if newStageFlag == False:
                    print("Stage : " + str(stage))
                    newStageFlag = True
                (success, box) = tracker.update(frame)

                if changeFlag == False:
                    st = time.time()
                    changeFlag = True

                ed = time.time()

                if ed - st >= 1:
                    centers = drawCircle(centers)
                    changeFlag = False

                idx = 0

                for center in centers:
                    if center[2] != 1:
                        cv2.circle(frame, (center[0], center[1]), 20, colors[idx], -1)
                    idx += 1

                if success:
                    (x, y, w, h) = [int(v) for v in box]

                    for center in centers:
                        if center[0] >= x and center[0] <= x+w and center[1] >= y and center[1] <= y + h and center[2] == 0:
                            center[2] = 1
                            score += 1

                    if score == target_score:
                        print("Congratulations!! You Clear Stage " + str(stage) + "!!")
                        score = 0
                        newStageFlag = False
                        stage += 1
                        if stage > 4:
                            stage = 1
                        centers, colors, target_score, stage = searchStage(stage)


                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "Score : " + str(score)
                    cv2.putText(frame, text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

                fps.update()
                fps.stop()

                info = [
                    ("Tracker", "kcf"),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                    ("Stage", str(stage))
                ]

                if success:
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Missing Object...", (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            else:
                cv2.putText(frame, "Press  'S' Button!!!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8)



            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                ret, frame = cam.read()
                frame = imutils.resize(frame, width=1000)
                cv2.putText(frame, "Select Your Face!!!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

                tracker.init(frame, initBB)
                fps = FPS().start()
            if key == ord("q"):
                break

        else:
            print("Video error!")
            break



