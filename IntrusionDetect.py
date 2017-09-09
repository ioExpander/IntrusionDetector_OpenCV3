import sys, os
import argparse
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
ap.add_argument("-c", "--conf", required=False, default="conf-intrusion.json", help="path to the JSON configuration file")
args = vars(ap.parse_args())

conf = json.load(open(args["conf"]))


if  conf["show_video"]:
    originalWindowName = "Original"
    processedWindowName = "Movement Indicator"
    cv2.namedWindow(processedWindowName)
    cv2.namedWindow(originalWindowName)
    cv2.moveWindow(originalWindowName, 800,20)

def diffImg(newFrame, runningAvg):
    d1 = cv2.absdiff(newFrame, cv2.convertScaleAbs(runningAvg))
    thresh = cv2.threshold(d1, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)
    return dilated
    #d2 = cv2.absdiff(t1, t0)
    #return cv2.bitwise_and(d1, d2)


def drawContours(diffFrame, origFrame):
    (_, cnts, _) = cv2.findContours(diffFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motionDetected = False
    maxDrawnContours = conf["maxDrawnContours"]
    drawnContours = 0
    # loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            #skip this contour as it is too small
            continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = (a*2 for a in cv2.boundingRect(c))
        cv2.rectangle(origFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        drawnContours +=1
        if drawnContours > maxDrawnContours:
            logger.debug("max drawn countours reached : {}".format(maxDrawnContours))
            break
    if drawnContours > 0:
        motionDetected = True
    return motionDetected

def reduceFrame(origFrame):
    reducedFrame = cv2.cvtColor(origFrame, cv2.COLOR_RGB2GRAY)
    reducedFrame = cv2.resize(reducedFrame, (0,0), fx=0.5, fy=0.5)
    reducedFrame = cv2.GaussianBlur(reducedFrame, (21, 21), 0)
    return reducedFrame


def initVideoCapture():
    cap = cv2.VideoCapture(conf["video_capture_source"])
    ret, readFrame = cap.read()
    if ret:
        # starting background model...
        avg = reduceFrame(readFrame).astype("float")
        logger.info("Video Monitoring Started")
        return ret, cap, avg
    else:
        logger.error("Error while initializing stream")
        return ret, None, None

def initVideoCaptureOrExit():
    ret, cap, avg = initVideoCapture();
    if not ret:
        logger.error("Unable to open stream")
        #Failed to open stream - Quit script
        sys.exit(-1)
    else:
        return cap, avg

def readInputImage():
    ret, readFrame = cap.read()
    if ret:
        fullFrame = readFrame
        reducedFrame = reduceFrame(fullFrame)
        cv2.accumulateWeighted(reducedFrame, avg, 0.5)
        return ret, fullFrame, reducedFrame
    else:
        logger.error("Error reading Frame")
        return ret, None, None

def uploadFileToDropbox(fileName, destPath):
    dbx=dropbox.Dropbox(conf["dropbox_token"])
    try:
        file_path = os.path.join('./', fileName)
        logger.info("Uploading {} to {}".format(file_path, destPath))
        with open(file_path, "rb") as f:
            dbx.files_upload(f.read(), destPath, mute=False)
    except Exception as err:
        logger.error("Failed to upload %s\n%s" % (file_path, err))

def uploadTrippedFileToDropbox(fileName):
    timestamp = datetime.datetime.now()
    tsDate = timestamp.strftime("%Y%m%d")
    dest_path = "/{}/{}".format(tsDate, fileName)
    uploadFileToDropbox(fileName, dest_path)

def uploadHealthCheckToDropbox(fileName):
    timestamp = datetime.datetime.now()
    tsDate = timestamp.strftime("%Y%m%d")
    dest_path = "/{}/Healthcheck/{}".format(tsDate, fileName)
    uploadFileToDropbox(fileName, dest_path)

#Init Video Source
cap, avg = initVideoCaptureOrExit();

# Read first image and fake buffer
ret, fullFrame, reducedFrame = readInputImage()
if not ret:
    logger.error("Error reading first image")
    #Failed to initialize - Quit script
    sys.exit(-1)
previousFrame = fullFrame

# init motion counter
motionFrameCount = 0
noMotionFrameCount = 0
tripped = False
recordedFrames = 0
recordedVideos = 0
framesSinceLastHealthCheck = conf["number_of_frames_before_healthcheck"]
errorFrames = 0
logger.info("System Initialized !")

while(True):
    timestamp = datetime.datetime.now()

    #Only draw contours and look for motion if not tripped or in show video mode
    if (not tripped) or (conf["show_video"]):
        diffFrame = diffImg(reducedFrame, avg)
        motionDetected = drawContours(diffFrame, fullFrame)
        if not motionDetected:
            noMotionFrameCount+=1
            #If no motion detected for too long reset motion counter
            if noMotionFrameCount >= conf["min_no_motion_frames"]:
                motionFrameCount = 0
                noMotionFrameCount = 0
        else: #Motion was detected !
            #reset No Motion Counter and increment motion Counter
            noMotionFrameCount = 0
            motionFrameCount+=1
            if motionFrameCount >= conf["min_motion_frames"]:
                tripped = True
                motionFrameCount = 0
                noMotionFrameCount = 0

    cv2.putText(fullFrame, "Tripped: {}, MotionFrameCount: {}, NoMotionFrameCount: {}".format(tripped, motionFrameCount, noMotionFrameCount), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
    cv2.putText(fullFrame, ts, (10, fullFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if (not tripped) and framesSinceLastHealthCheck >= conf["number_of_frames_before_healthcheck"]:
        framesSinceLastHealthCheck = 0
        tsShort = timestamp.strftime("%Y%m%d-%Hh%Mm%S")
        hcImgName = 'hc{}.jpg'.format(tsShort)
        logger.info("Healthcheck Time : {}".format(hcImgName))
        cv2.imwrite(hcImgName, fullFrame)
        if conf["use_dropbox"]:
            uploadHealthCheckToDropbox(hcImgName)

    if conf["show_video"]:
        cv2.imshow(processedWindowName,diffFrame)
        cv2.imshow(originalWindowName,fullFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if tripped:
        if recordedFrames == 0:
            tsShort = timestamp.strftime("%Y%m%d-%Hh%Mm%S")

            outImgName = 'out{}.jpg'.format(tsShort)
            logger.warning("Tripped. Saving image {}".format(outImgName))
            cv2.imwrite(outImgName, fullFrame)
            #upload to Dbx
            if conf["use_dropbox"]:
                uploadTrippedFileToDropbox(outImgName)

            outVideoName = 'out{}.avi'.format(tsShort)
            logger.warning("Tripped. Recording {}".format(outVideoName))
            out = cv2.VideoWriter(outVideoName,cv2.VideoWriter_fourcc('X','V','I','D'), 10.0, (fullFrame.shape[1],fullFrame.shape[0]))
            #write previous Frame in video.
            out.write(previousFrame)
        #Write current frame in video
        out.write(fullFrame)
        recordedFrames+=1
        if recordedFrames >= conf["frames_to_record"]:
            tripped = False
            recordedFrames = 0
            recordedVideos+=1
            out.release()
            logger.info("Stopping Recording - Video #{}".format(recordedVideos))
            if conf["use_dropbox"]:
                #Pause capture
                cap.release()
                #upload to Dbx
                uploadTrippedFileToDropbox(outVideoName)
                #Restart capture
                cap, avg = initVideoCaptureOrExit();

            if recordedVideos >= conf["max_videos_to_make"]:
                logger.warning("Max Videos Reached {} - Stopping".format(recordedVideos))
                break


    # Read next frame
    previousFrame = fullFrame;
    ret, readFullFrame, readReducedFrame = readInputImage()
    if ret:
        fullFrame = readFullFrame
        reducedFrame = readReducedFrame
        #increment frame counter for Healthcheck
        framesSinceLastHealthCheck+=1
        errorFrames = 0
    else:
        errorFrames+=1
        if errorFrames >= conf["max_error_frames"]:
            logger.error("Too many errors in imput stream")
            sys.exit(-1)


# When everything done, release the capture
cap.release()
if conf["show_video"]:
    #cv2.destroyWindow(originalWindowName) #not needed when using destroyAllWindows
    #cv2.destroyWindow(processedWindowName)
    cv2.destroyAllWindows()
