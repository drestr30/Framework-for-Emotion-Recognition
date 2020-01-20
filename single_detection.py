import cv2 as cv
from detection import Detection
import numpy as np
import time
import Filtro
import  matplotlib.pyplot as plt

start = time.time()

### input image
path = '/Users/david/Desktop/Grimmanet/Images/MUG/Neutral.jpg'

#initialize detector
detector = Detection()
detector.load_cascades()
detector.load_CNN_detector()

img = cv.imread(path)
# bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = detector.filter_homo(img)
cv.imshow('img', gray)
cv.waitKey(0)
cv.destroyAllWindows()

faces = detector.CNN_face_detection(img, 0.8)
(x,y,w,h) = list(map(int, faces[0]))
cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
roi_gray = gray[y:y+h, x:x+w]
roi_color = img[y:y+h, x:x+w]

#Eyes detection

eyes_roiBB = [0, int(h/6), w, int(h/2)]
# cv.rectangle(roi_gray, (eyes_roiBB[0], eyes_roiBB[1]), (eyes_roiBB[0]+eyes_roiBB[2], eyes_roiBB[1]+eyes_roiBB[3]), (123, 123, 123), 2)
roi_eyes = roi_gray[eyes_roiBB[1]:eyes_roiBB[1]+eyes_roiBB[3], eyes_roiBB[0]:eyes_roiBB[0]+eyes_roiBB[2]]
eyes = detector.eyes_detection(roi_eyes)
# for (ex,ey,ew,eh) in eyes:
#     cv.rectangle(roi_eyes,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
    # seg_
    # img[eyes_roiBB[1]+ey:(eyes_roiBB[1]+ey+eh), (eyes_roiBB[0]+ex):(eyes_roiBB[0]+ex+ew)] = roi_eyes[ey:ey+eh, ex:ex+ew]

#Mouth detection

smile_roiBB = [int(w / 6), int(h / 2), int(4* w / 6), int(h / 2)] #[int(w/4), int(2*h/3), int(w/2), int(h/3)]
# cv.rectangle(roi_gray, (smile_roiBB[0], smile_roiBB[1]), (smile_roiBB[0]+smile_roiBB[2], smile_roiBB[1]+smile_roiBB[3]), (123, 123, 123), 2)
roi_smile = roi_gray[smile_roiBB[1]:smile_roiBB[1]+smile_roiBB[3], smile_roiBB[0]:smile_roiBB[0]+smile_roiBB[2]]
smile = detector.mouth_detection(roi_smile)# , (x,y,w,h))
# for (sx,sy,sw,sh) in smile:
#     cv.rectangle(roi_smile, (sx,sy), (sx+sw,sy+sh), (0, 0, 255), 2)
    # seg_img[(smile_roiBB[1]+sy):(smile_roiBB[1]+sy)+sh, smile_roiBB[0]+sx:smile_roiBB[0]+sx+sw] = roi_smile[sy:sy + sh, sx:sx + sw]

#Nose detection
#
# nose_roiBB = [int(w/3), int(2*h/4), int(w/3), int(h/4)]
# # cv.rectangle(roi_color, (nose_roiBB[0], nose_roiBB[1]),(nose_roiBB[0] + nose_roiBB[2], nose_roiBB[1] + nose_roiBB[3]), (0, 255, 120), 2)
# roi_nose = roi_gray[nose_roiBB[1]:nose_roiBB[1]+nose_roiBB[3],nose_roiBB[0]:nose_roiBB[0]+nose_roiBB[2]]
# #  nose = nose_cascade.detectMultiScale(roi_gray, minSize= (int(w/5.5), int(h/5.5)), maxSize= (int(w/4), int(h/4)))
# nose = detector.nose_detection(roi_nose, (x,y,w,h))
# for (nx,ny,nw,nh) in nose:
    # cv.rectangle(roi_nose, (nx, ny), (nx + nw, ny + nh), (255, 255, 255), 2)q
    # seg_img[nose_roiBB[1]+ny:nose_roiBB[1]+ny + nh, nose_roiBB[0]+nx:nose_roiBB[0]+nx + nw] = roi_nose[ny:ny + nh, nx:nx + nw]
cv.imwrite('./roy_gray.jpg', roi_gray)
# cv.imshow('img', roi_gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

if not isinstance(smile, tuple) and not isinstance(eyes, tuple):

    smile_max_index = np.where(smile[:, 2] == smile[:, 2].max())[0][0]
    (x1r, y1r) = [smile_roiBB[0] + smile[smile_max_index, 0] + smile[smile_max_index, 2],
                  smile_roiBB[1] + smile[smile_max_index, 1] + smile[smile_max_index, 3]]
    (x1l, y1l) = [smile_roiBB[0] + smile[smile_max_index, 0],
                  smile_roiBB[1] + smile[smile_max_index, 1] + smile[smile_max_index, 3]]
    try:
        eye_right_index = np.where(eyes[:, 0] == eyes[:, 0].max())[0][0]
        (_x2r, _y2r) = [eyes_roiBB[0] + eyes[eye_right_index, 0] + eyes[eye_right_index, 2],
                        eyes_roiBB[1] + eyes[eye_right_index, 1] + eyes[eye_right_index, 3]]
        y3r = eyes_roiBB[1] + eyes[eye_right_index, 1] - eyes[eye_right_index, 3]
        x3r = (eyes[0, 1] - y1r) * (_x2r - x1r) / (_y2r - y1r) + x1r
    except IndexError:
        (x3r, y3r) = (0, 0)

    try:
        eye_left_index = np.where(eyes[:, 0] == eyes[:, 0].min())[0][0]
        (_x2l, _y2l) = [eyes[eye_left_index, 0],
                        eyes_roiBB[1] + eyes[eye_left_index, 1] + eyes[eye_left_index, 3]]
        y3l = eyes_roiBB[1] + eyes[eye_left_index, 1] - eyes[eye_left_index, 3]
        x3l = (eyes[1, 1] - y1l) * (_x2l - x1l) / (_y2l - y1l) + x1l
    except IndexError:
        (x3l, y3l) = (0, 0)

    if x3r == 0 or x3r < w / 2:
        (x3r, y3r) = (w - x3l, y3l)

    if x3l == 0 or x3l > w / 2:
        (x3l, y3l) = (w - x3r, y3r)

    vertices = np.array([[x1r, y1r], [x1l, y1l], [x3r, y3r], [x3l, y3l]], dtype=np.int32)
    vertices = cv.convexHull(vertices)

    # Mask image
    mask_img = np.zeros_like(roi_gray)
    cv.fillConvexPoly(mask_img, vertices, 255)

    cv.imwrite('./mask.jpg', mask_img)
    # cv.imshow('img', mask_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # LG filter stage
    width, height = roi_gray.shape

    kernel = Filtro.laguerre_gauss_filter(height, width, 0.8)
    lenna_fourier = Filtro.fourier_transform(roi_gray)
    kernel_fourier = Filtro.fourier_transform(kernel)

    out = np.multiply(kernel_fourier, lenna_fourier)
    out = np.abs(Filtro.inverse_fourier_transform(out))

    fil_img = out / out.max()  # normalize the data to 0 - 1
    fil_img = 255 - (255 * fil_img) # Now scal by 255
    fil_img = fil_img.astype(np.uint8)
    cv.imwrite('./filtered_mug.jpg', fil_img)
    # cv.imshow("filtered", fil_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    masked = cv.bitwise_and(fil_img, mask_img)

    cv.imwrite('./segmented_image.jpg', masked)
    # cv.imshow('img', masked)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # cv.imwrite('/User/david/Desktop', fil_img)




        # #Eyebrow segmentation
        # (x1,y1) = [int(nose[0,0]+nose[0,2]/2), nose[0,1]+int(nose[0,3]/2)]
        #
        #
        # if eyes[1,0] > eyes[0,1]:
        #     (x2r,y2r) = [eyes[1,0]+eyes[1,2], int(eyes[1,1]+eyes[1,3]/2)]
        #     p3r = (eyes[1,1] - y1)*(x2r-x1)/(y2r-y1)+x1
        #     (x3r,y3r, wr, hr) = [eyes[1,0], eyes[1,1]- int(eyes[1,3]/2), int(p3r)-eyes[1,0], int(eyes[1,3]/2)]
        #
        #     (x2l, y2l) = [eyes[0, 0], int(eyes[0, 1] + eyes[0, 3] / 2)]
        #     p3l = (eyes[0, 1] - y1) * (x2l - x1) / (y2l - y1) + x1
        #     (x3l, y3l, wl, hl) = [int(p3l), eyes[0, 1] - int(eyes[0, 3] / 2), int(eyes[0, 0]+eyes[0,2]-p3l), int(eyes[0,3]/2)]
        #
        # else:
        #     (x2r, y2r) = [eyes[0, 0] + eyes[0, 2], int(eyes[0, 1] + eyes[0, 3] / 2)]
        #     p3r = (eyes[1, 1] - y1) * (x2r - x1) / (y2r - y1) + x1
        #     (x3r, y3r, wr, hr) = [eyes[0, 0], eyes[0, 1] - int(eyes[0, 3] / 2), int(p3r) - eyes[0, 0], int(eyes[0, 3] / 2)]
        #
        #     (x2l, y2l) = [eyes[1, 0], int(eyes[1, 1] + eyes[1, 3] / 2)]
        #     p3l = (eyes[1, 1] - y1) * (x2l - x1) / (y2l - y1) + x1
        #     (x3l, y3l, wl, hl) = [int(p3l), eyes[1, 1] - int(eyes[1, 3] / 2), int(eyes[1, 0] + eyes[1, 2] - p3l),
        #                           int(eyes[1, 3] / 2)]
        #
        #
        # cv.rectangle(roi_color, (x3r, y3r), (x3r + wr, y3r + hr), (255, 255, 255), 2)
        # cv.rectangle(roi_color, (x3l, y3l), (x3l + wl, y3l + hl), (255, 255, 255), 2)
        # seg_img[y3r:y3r + hr, x3r:x3r + wr] = roi_color[y3r:y3r + hr, x3r:x3r + wr]
        # seg_img[y3l:y3l + hl, x3l:x3l + wl] = roi_color[y3l:y3l + hl, x3l:x3l + wl]
        # cv.imshow('img', seg_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()



# end = time.time()
# print(end - start)


######
# Emotion detection
#
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.mobilenet import preprocess_input
#
# #
# #
# model = load_model('/Users/david/Desktop/Grimmanet/saved_models/emotion_model_20ep.h5')
# labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
# # print(model.summary())
#
# new_img = cv.resize(masked, (224,224), interpolation=cv.INTER_CUBIC)
# cv.imshow('new image', new_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# image = np.zeros((224,224,3))
# image[:,:,0] = new_img
# image[:,:,1] = new_img
# image[:,:,2] = new_img
#
#
# image = np.expand_dims(image, 0)
#
# pred = model.predict(preprocess_input(image))
# pred_coded = np.argmax(pred, axis=1)
# prediction = labels[pred_coded[0]]
# print(prediction)