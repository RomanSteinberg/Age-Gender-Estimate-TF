import numpy as np
import cv2
from collections import OrderedDict

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

class FaceAligner:
    def __init__(self, config, predictor, desiredLeftEye=(0.35, 0.35)):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.config = config
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye

    def get_resized_image(self, image, aligned_face, M, x_indent, y_indent):
        init_coords = [[x_indent, y_indent], [image.shape[0] + x_indent, y_indent],
                       [x_indent, image.shape[1] + y_indent], [image.shape[0] + x_indent, image.shape[1] + y_indent]]
        transf_coords = []
        for pair in init_coords:
            x, y = pair
            x_1 = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            y_1 = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            transf_coords.append([int(x_1), int(y_1)])
        arr = np.asarray(transf_coords)
        x_max = arr[:, :1].max()
        if x_max > image.shape[0]+2*x_indent:
            x_max = image.shape[0]+2*x_indent
        x_min = arr[:, :1].min()
        if x_min < 0:
            x_min = 0
        y_max = arr[:, 1:2].max()
        if y_max > image.shape[1]+2*y_indent:
            y_max = image.shape[1]+2*y_indent
        y_min = arr[:, 1:2].min()
        if y_min < 0:
            y_min = 0
        cropped_face = aligned_face[y_min:y_max, x_min:x_max, :]
        image_size = self.config['size']
        resized_face = cv2.resize(cropped_face, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        return resized_face

    def align(self, image, gray, rect):
        self.desiredFaceWidth = int(2 * image.shape[0])
        self.desiredFaceHeight = int(2 * image.shape[1])
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        x_indent = int(scale * image.shape[0] / 2)
        y_indent = int(scale * image.shape[1] / 2)

        bordered_image = cv2.copyMakeBorder(image, y_indent, y_indent, x_indent, x_indent, cv2.BORDER_CONSTANT)
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2 + x_indent,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2 + y_indent)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale=1)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])


        # apply the affine transformation
        aligned_face = cv2.warpAffine(bordered_image, M, (int(self.desiredFaceWidth), int(self.desiredFaceHeight)),
                                flags=cv2.INTER_LANCZOS4)
        return self.get_resized_image(image, aligned_face, M, x_indent, y_indent)
