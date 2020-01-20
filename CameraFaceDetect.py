import cv2 as cv
from detection import Detection
import time
from imutils.video import VideoStream, FileVideoStream
import imutils
import numpy as np
import Filtro
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
# from keras.applications.inception_v3 import preprocess_input
import threading
import tensorflow as tf


class InferenceThread(threading.Thread):
    def __init__(self, model_, frame_, labels_):
        threading.Thread.__init__(self)
        self.name = "inference"
        self.model = model_
        self.frame = frame_
        self.labels = labels_
        self._return = None

    def run(self):
        # print("Starting" + self.name)
        self._return = self.predict_emotion()

    def predict_emotion(self):

        img_size = (48,48)   #(224, 224) for mobilNet based model

        new_img = cv.resize(self.frame, img_size, interpolation=cv.INTER_CUBIC)
        emo_image = np.zeros((48, 48, 1))
        emo_image[:, :, 0] = new_img
        # emo_image[:, :, 1] = new_img
        # emo_image[:, :, 2] = new_img

        with graph.as_default():
            emo = self.model.predict(preprocess_input(np.expand_dims(emo_image, 0)))
            return emo

    def join(self):
        threading.Thread.join(self)
        return self._return


global graph, start_time, vertices

model = load_model('/Users/david/Desktop/Grimmanet/saved_models/mini_XCEPT.FER.65-0.62.hdf5', compile= False)
graph = tf.get_default_graph()
labels = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
# print(model.summary())

start_time = time.time()
one_seg = 1 # displays the frame rate every 1 second
counter = 0

detector = Detection()
detector.load_cascades()
detector.load_CNN_detector()

# initialize the video stream and allow the cammera sensor to warmup
# print("[INFO] starting video stream...")
# video_capture = VideoStream(src=0).start()
# time.sleep(2.0)

##### initialize video from file
video_path = "/Users/david/Desktop/Grimmanet/VideosClaseAI/webex.mp4"
print("[INFO] starting video file thread...")
video_capture = FileVideoStream(video_path).start()
time.sleep(2.0)

while True:

    # Capture frame-by-frame
    frame = video_capture.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=600)

    try:
        face = detector.CNN_face_detection(frame, 0.8)
    except ValueError:
        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Draw a rectangle around the faces
    (x, y, w, h) = face
    roi_face = frame[y: y+h, x: x+w]
    roi_gray = cv.cvtColor(roi_face, cv.COLOR_BGR2GRAY)

    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Detect eyes and mouth in the detected faces
    eyes_roiBB = [0, int(h / 4), w, int(h / 4)]
    roi_eyes = roi_face[eyes_roiBB[1]:eyes_roiBB[1] + eyes_roiBB[3], eyes_roiBB[0]:eyes_roiBB[0] + eyes_roiBB[2]]
    eyes = detector.eyes_detection(roi_eyes)
    # for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_eyes, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    smile_roiBB = [int(w / 4), int(2 * h / 3), int(w / 2), int(h / 3)]
    roi_smile = roi_face[smile_roiBB[1]:smile_roiBB[1] + smile_roiBB[3],
                    smile_roiBB[0]:smile_roiBB[0] + smile_roiBB[2]]
    smile = detector.mouth_detection(roi_smile, (x, y, w, h))
    # for (mx, my, mw, mh) in mouth:
        # cv2.rectangle(roi_smile, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

    # Create the mask image to segment faces
    mask = np.zeros_like(roi_gray)
    try:
        global vertices
        vertices = detector.create_segment_mask(face, smile, smile_roiBB, eyes, eyes_roiBB)
        print("Segment mask actualized")

    except TypeError:
        print("Segment mask not actualized, not eyes and mouth detected")
        pass

    try:
        cv.fillConvexPoly(mask, vertices, 255)
    except NameError:
        print("No vertices recognized, eyes and mouth not detected ")
        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # LG filter stage
    width, height = roi_gray.shape

    kernel = Filtro.laguerre_gauss_filter(height, width, 0.8)
    lenna_fourier = Filtro.fourier_transform(roi_gray)
    kernel_fourier = Filtro.fourier_transform(kernel)

    out = np.multiply(kernel_fourier, lenna_fourier)
    out = np.abs(Filtro.inverse_fourier_transform(out))

    fil_img = out / out.max()  # normalize the data to 0 - 1
    fil_img = 255 * fil_img  # Now scal by 255
    fil_img = fil_img.astype(np.uint8)

    masked = cv.bitwise_and(fil_img, mask)  # segmented and filtered image

    #Start emotion recognition thread
    inference = InferenceThread(model, masked, labels)
    inference.start()

    #Prepare image to display in subwindow.
    masked = imutils.resize(masked, width=100)
    frame[0:masked.shape[0], frame.shape[1] - masked.shape[1]:frame.shape[1], 0] = masked
    frame[0:masked.shape[0], frame.shape[1] - masked.shape[1]:frame.shape[1], 1] = masked
    frame[0:masked.shape[0], frame.shape[1] - masked.shape[1]:frame.shape[1], 2] = masked

    prediction = inference.join()



    canvas = np.zeros((250, 300, 3), dtype="uint8")

    for (i, (emotion, prob)) in enumerate(zip(labels, prediction)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        cv.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv.putText(canvas, text, (10, (i * 35) + 23),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)

    counter += 1

    if time.time() - start_time > one_seg:
        print("FPS: ", counter / (time.time() - start_time), "frames per second")
        counter = 0
        start_time = time.time()

    cv.imshow('Video', frame)
    cv.imshow("Probabilities", canvas)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.stop()
cv.destroyAllWindows()