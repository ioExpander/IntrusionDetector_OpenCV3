import sys, os
import argparse
import time
import logging
import json
import datetime
import numpy as np
import cv2
import dropbox


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=False, default="conf-garden.json", help="path to the JSON configuration file")
args = vars(ap.parse_args())

conf = json.load(open(args["conf"]))


if  conf["show_video"]:
    originalWindowName = "Original"
    cv2.namedWindow(originalWindowName)
    cv2.moveWindow(originalWindowName, 800,20)

def reduceFrame(origFrame):
    reducedFrame = cv2.cvtColor(origFrame, cv2.COLOR_RGB2GRAY)
    reducedFrame = cv2.resize(reducedFrame, (0,0), fx=0.5, fy=0.5)
    reducedFrame = cv2.GaussianBlur(reducedFrame, (21, 21), 0)
    return reducedFrame


def initVideoCapture():
    for i in range(0, conf["max_reconnects"]):
        logger.info("Stream connection attempt #{}".format(i))
        cap = cv2.VideoCapture(conf["video_capture_source"])
        if not (cap is None):
            ret, readFrame = cap.read()
            if ret:
                # starting background model...
                logger.info("Video Monitoring Started")
                return ret, cap
            else:
                logger.error("Error while initializing stream")
                return ret, cap
        else:
            #Wait a little before retrying
            time.sleep(10)
    #Failed to establish TCP connection
    logger.error("Could Not connect to Camera - Exiting")
    sys.exit(-1)

def initVideoCaptureOrExit():
    for i in range(0, conf["max_reconnects"]):
        ret, cap = initVideoCapture();
        if ret:
            return cap
        else:
            #Wait a little before retrying
            time.sleep(10)

    #Failed to open stream - Quit script
    logger.error("Unable to read first image - Exiting")
    sys.exit(-1)

def readInputImage():
    ret, readFrame = cap.read()
    if ret:
        fullFrame = readFrame
        return ret, fullFrame
    else:
        logger.error("Error reading Frame")
        return ret, None

def uploadFileToDropbox(fileName, destPath):
    dbx=dropbox.Dropbox(conf["dropbox_token"])
    try:
        file_path = os.path.join('./', fileName)
        logger.info("Uploading {} to {}".format(file_path, destPath))
        with open(file_path, "rb") as f:
            dbx.files_upload(f.read(), destPath, mute=False)
    except Exception as err:
        logger.error("Failed to upload %s\n%s" % (file_path, err))

def uploadTimelapseFileToDropbox(fileName):
    timestamp = datetime.datetime.now()
    tsDate = timestamp.strftime("%Y%m%d")
    dest_path = "/{}/{}".format(tsDate, fileName)
    uploadFileToDropbox(fileName, dest_path)

logger.info("====  STARTING  ====")

#Init Video Source
cap = initVideoCaptureOrExit();

# Read first image and fake buffer
ret, fullFrame = readInputImage()
if not ret:
    logger.error("Error reading first image")
    #Failed to initialize - Quit script
    sys.exit(-1)

# init motion counter
recordedVideos = 0
errorFrames = 0
reconnectAttempts = 0
shouldExit=False

logger.info("System Initialized !")

while(not shouldExit):
    timestamp = datetime.datetime.now()
    tsShort = timestamp.strftime("%Y%m%d-%Hh%Mm%S")
    outVideoName = 'lapse{}.avi'.format(tsShort)
    logger.warning("Starting recording : {}".format(outVideoName))
    out = cv2.VideoWriter(outVideoName,cv2.VideoWriter_fourcc('X','V','I','D'), 5.0, (fullFrame.shape[1],fullFrame.shape[0]))
    recordedFrames = 0
    for i in range(0, conf["max_frames_per_vid"]):
        # Read frame
        ret, readFullFrame = readInputImage()
        if ret:
            fullFrame = readFullFrame
            #increment frame counter for Healthcheck
            errorFrames = 0
            reconnectAttempts = 0
        else:
            errorFrames+=1
            if errorFrames >= conf["max_error_frames"]:
                logger.error("Too many errors in imput stream")
                if reconnectAttempts > conf["max_reconnects"]:
                    shouldExit=True
                    break
                else:
                    reconnectAttempts+=1
                    logger.error("Reconnecting stream - Attempt #{}".format(reconnectAttempts))
                    ret, cap = initVideoCapture()
                    errorFrames=0


        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        cv2.putText(fullFrame, ts, (10, fullFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if conf["show_video"]:
            cv2.imshow(originalWindowName,fullFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shouldExit=True
                break
        #write current frame in video
        out.write(fullFrame)
        recordedFrames+=1

        #wait for next frame
        time.sleep(conf["wait_seconds_between_caps"])

    logger.info("Stopping Recording - Video #{}".format(recordedVideos))
    out.release()
    recordedVideos+=1

    if conf["use_dropbox"]:
        #Pause capture
        cap.release()
        #upload to Dbx
        uploadTimelapseFileToDropbox(outVideoName)
        #Restart capture
        if not shouldExit:
            cap = initVideoCaptureOrExit();

# When everything done, release the capture
cap.release()
if conf["show_video"]:
    cv2.destroyAllWindows()
logger.info("Exiting program")
