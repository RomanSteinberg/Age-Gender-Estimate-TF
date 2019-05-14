import os
import cv2
import dlib
import json
import yaml
from tqdm import tqdm
from imutils.face_utils import FaceAligner
from age_gender.utils.preprocess import align_face, get_area, scale


class Converter:

    def __init__(self, config):
        self.config = config
        self.image_size = config['image']['size']
        self.json = config['general']['json_path']

        self.shape_predictor = 'shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.face_aligner = FaceAligner(self.predictor, desiredFaceWidth=self.image_size)

    def convert_dataset(self):
        dataset_path = self.config['general']['dataset_path']
        processed_dataset_path = self.config['general']['processed_dataset_path']
        with open(self.json) as f:
            dataset = json.load(f)

        for record in tqdm(dataset, bar_format='Progress {bar} {percentage:3.0f}% [{elapsed}<{remaining}'):
            file_name = record['file_name'][0]
            image_path = os.path.join(dataset_path, file_name)
            save_path = os.path.join(processed_dataset_path, file_name)
            if os.path.exists(save_path):
                continue
            processed_image = self.convert_image(image_path)
            if processed_image is not None:
                save_folder_path = os.path.dirname(os.path.abspath(save_path))
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                cv2.imwrite(save_path, processed_image)
        self.save_dataset_config(processed_dataset_path)

    def convert_image(self, image_path):
        face_area_threshold = self.config['image']['face_area_threshold']

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        w, h = image.shape[:2]
        rects = self.detector(image, 2)
        if len(rects) != 1:
            return None
        else:
            rect = rects[0]
            area = get_area(rect)
            if area < face_area_threshold:
                return None

            # align
            face_aligner = self.face_aligner
            if self._is_like_image(image.shape):
                face_aligner = FaceAligner(self.predictor, desiredFaceWidth=w, desiredFaceHeight=h)
            aligned_face = align_face(self.config['image'], rect, face_aligner, image, gray_image)

            # crop
            rect = self.detector(image, 2)[0]
            rect = scale(self.config['image'], rect)
            face = aligned_face[rect[1]:rect[3], rect[0]:rect[2], :]

            # resize
            if self._is_like_image(aligned_face.shape):
                face = cv2.resize(face, (self.image_size, self.image_size))
            return face

    def _is_like_image(self, shape):
        size = shape[0] != self.image_size or shape[1] != self.image_size
        return size

    def save_dataset_config(self, processed_dataset_path):
        with open(os.path.join(processed_dataset_path, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


