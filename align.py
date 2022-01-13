# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import numpy as np

def detect_image(image, output='uploads/output', zoom_factor=0.40, aspect_ratio='35:45', shape_predictor="shape_predictor_68_face_landmarks.dat", 
                 pixel_size=1800, ):
    # args = vars(ap.parse_args())
    zoom_factor = float(zoom_factor)
    pixel_size = int(pixel_size)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredLeftEye=(zoom_factor, zoom_factor),
                     desiredFaceWidth=pixel_size)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image)
    image = imutils.resize(image, width=pixel_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)
    ratio = list(map(int, aspect_ratio.split(':')))
    breedste = int(0)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)

        if (w > breedste):
            breedste = int(w)

            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=pixel_size)
            faceAligned = fa.align(image, gray, rect)

            # photo is cropped by the given aspect ratio
            if ratio[0] > ratio[1]:
                height = int(pixel_size / ratio[0] * ratio[1])
                diff = int((pixel_size - height) / 2)
                faceAligned = faceAligned[diff:height + diff, :, :]
            elif ratio[0] < ratio[1]:
                width = int(pixel_size / ratio[1] * ratio[0])
                diff = int((pixel_size - width) / 2)
                faceAligned = faceAligned[:, diff:width + diff, :]

            # saves image as jpg file if image name has been entered
            if output is not None:
                cv2.imwrite(output + ".JPG", faceAligned)

