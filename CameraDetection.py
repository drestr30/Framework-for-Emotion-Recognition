import cv2
import imutils
from imutils.video import VideoStream, FileVideoStream
from imutils.video import FPS
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import mobilenet
from detection import Detection
import time
import Filtro
import matplotlib.pyplot as plt
# import keyboard
# import seaborn as sns

global  start_time, vertices, emotion_report
# Time for displaying FPS in seconds
one_seg = 5 # displays the frame rate every second

# Face and facial feature detectors.
detector = Detection()
detector.load_cascades()
detector.load_CNN_detector()

# Emotion model path
emotion_model_path = './EmotionModels/MxModels/mXC.ferBalance.16-0.61.hdf5' #MixBestModel.h5'
model_input_size = (48,48)

# hyper-parameters for bounding boxes shape
# loading models
# segmentate = False
report_emotion = []
report_time = []

video_source = 'webcam'   ## OPTIONS 'webcam' and 'video_file'
file_name = 'AIClassAugmentation'
video_path = './TestVideos/%s.mp4'%file_name # if video_file is selected
detect_emotion = True
apply_filter = False
display_roi = False
save_procesed_video = False

emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]  #for Xception in FER-DB####
# EMOTIONS = ["angry", "disgust", "scared", "sad"]
#EMOTIONS = ["angry", "disgust", "scared", "sad",'neutral']
###['angry', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

#####

if video_source == 'webcam':
    #starting video streaming
    print("[INFO] starting video stream...")
    video_capture = VideoStream(src=0).start()
    time.sleep(2.0)
elif video_source == 'video_file':
    # recognition form video file
    print("[INFO] starting video file thread...")
    video_capture = FileVideoStream(video_path).start()
    time.sleep(2.0)
#
# class EmotionReport():
#
#     def __init__(self):
#         self.emotions = []
#         self.time = []
#
#     def add2report(self, emotion):
#         self.emotions.append(emotion)
#         fps.stop()
#         self.time.append(fps.elapsed())
#
#     def plot_report(self):
#         plt.plot(self.emotions, self.time)

if save_procesed_video:
    out = cv2.VideoWriter('./processed_videos/%s_processed.avi'%file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (594, 250)) #'M', 'J', 'P', 'G'

def preprocess_input(x, v2=True, f=False):
    x = x.astype('float32')

    if f:
        x = (255 - x) / 255.0
    else :
        x = x / 255.0

    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def filter_images(roi):
    width, height = roi.shape[:2]
    kernel = Filtro.laguerre_gauss_filter(height, width, 0.7)

    lenna_fourier = Filtro.fourier_transform(roi)
    kernel_fourier = Filtro.fourier_transform(kernel)

    out = np.multiply(kernel_fourier, lenna_fourier)
    out = np.abs(Filtro.inverse_fourier_transform(out))

    fil_img = out / out.max()  # normalize the data to 0 - 1
    fil_img = 255 - 255 * fil_img  # Now scal by 255
    fil_img = fil_img.astype(np.uint8)

    return fil_img

fps = FPS().start()

while True:

    frame = video_capture.read()

    if video_capture.stopped:
        break
    # reading the frame
    # print(np.shape(frame))
    width, height, _ = np.shape(frame)
    frame = imutils.resize(frame, height=height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try: faces = detector.CNN_face_detection(frame, 0.7)
    except ValueError: faces = []; pass

    canvas = np.zeros((height, int(width/1.5), 3), dtype="uint8")

    frameClone = frame.copy()

    if faces.shape[0] >= 1:

        for face in faces:
            (x, y, w, h) = face.astype(int)
            # print(w, h)
            cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Extract the ROI of the face from the grayscale image, resize it to the fixed model size, and then prepare
            # the ROI for classification via the CNN
            roi = gray[y:y + h, x:x + w]
            roiColor = frameClone[y:y + h, x:x + w]

            if apply_filter:  # LG filter stage
                roi = filter_images(roi)

            if display_roi:
                roi_display = imutils.resize(roi, width=100)
                frameClone[0:roi_display.shape[0], frameClone.shape[1] - roi_display.shape[1]:frameClone.shape[1],
                0] = roi_display
                frameClone[0:roi_display.shape[0], frameClone.shape[1] - roi_display.shape[1]:frameClone.shape[1],
                1] = roi_display
                frameClone[0:roi_display.shape[0], frameClone.shape[1] - roi_display.shape[1]:frameClone.shape[1],
                2] = roi_display

            try: roi = cv2.resize(roi, model_input_size) # Resize roi to the model input size
            except cv2.error: continue
                # roi = roi.astype("float") / 255.0


            if model_input_size == (224,224):
                new_roi = np.zeros((224, 224, 3))

                new_roi[:, :, 0] = roi
                new_roi[:, :, 1] = roi
                new_roi[:, :, 2] = roi
                roi = mobilenet.preprocess_input(new_roi)

            if detect_emotion :

                roi = img_to_array(roi)
                roi = np.expand_dims(preprocess_input(roi, apply_filter), axis=0)

                preds = emotion_classifier.predict(roi)[0]  # send model to perform emotion recognition
                label = EMOTIONS[preds.argmax()]
                report_emotion.append(preds)
                fps.stop()
                report_time.append(fps.elapsed())
        # if segmentate:
        #     # Detect eyes and mouth in the detected faces
        #     eyes_roiBB = [0, int(h / 4), w, int(h / 4)]
        #     roi_eyes = roi[eyes_roiBB[1]:eyes_roiBB[1] + eyes_roiBB[3], eyes_roiBB[0]:eyes_roiBB[0] + eyes_roiBB[2]]
        #     eyes = detector.eyes_detection(roi_eyes)
        #     # for (ex, ey, ew, eh) in eyes:
        #     # cv2.rectangle(roi_eyes, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        #
        #     smile_roiBB = [int(w / 4), int(2 * h / 3), int(w / 2), int(h / 3)]
        #     roi_smile = roi[smile_roiBB[1]:smile_roiBB[1] + smile_roiBB[3],
        #                 smile_roiBB[0]:smile_roiBB[0] + smile_roiBB[2]]
        #     smile = detector.mouth_detection(roi_smile, (x, y, w, h))
        #     # for (mx, my, mw, mh) in mouth:
        #     # cv2.rectangle(roi_smile, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
        #
        #     # Create the mask image to segment faces
        #     mask = np.zeros_like(roi)
        #     try:
        #         global vertices
        #         vertices = detector.create_segment_mask(roi, smile, smile_roiBB, eyes, eyes_roiBB)
        #         print("Segment mask actualized")
        #
        #     except TypeError:
        #         print("Segment mask not actualized, not eyes and mouth detected")
        #         pass
        #
        #     try:
        #         cv2.fillConvexPoly(mask, vertices, 255)
        #     except NameError:
        #         print("No vertices recognized, eyes and mouth not detected ")
        #         pass

                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {}%".format(emotion, int(prob * 100))
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 70) + 20),
                                  (w, (i * 70) + 65), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 70) + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fps.update()
    fps.stop()

    procesed = np.hstack((frameClone, canvas))

    if save_procesed_video: out.write(procesed)

    resized = cv2.resize(procesed, (1870,1030))
    cv2.imshow('procesed', resized)
    # cv2.imshow('your_face', frameClone)
    # cv2.imshow("your_emotions", canvas)


    if int(fps.elapsed()) % one_seg == 0:
        #print("[INFO] approx. time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        fps.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        break

video_capture.stop()
# cv2.destroyAllWindows()
time.sleep(1.0)

emotion_array= np.asarray(report_emotion)
# sns.heatmap(np.transpose(emotion_array), yticklabels=EMOTIONS, xticklabels=False)
plt.show()
# plt.hea(report_time, report_emotion)
# plt.show()







