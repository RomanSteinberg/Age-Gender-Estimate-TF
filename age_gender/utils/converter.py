import os
import cv2
import dlib
import json
import yaml
from imutils.face_utils import FaceAligner
from age_gender.utils.preprocess import align_face, get_area


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
        for record in dataset:
            file_name = record['file_name'][0]
            image_path = os.path.join(dataset_path, file_name)
            processed_image = self.convert_image(image_path)
            if processed_image is not None:
                save_path =  os.path.join(processed_dataset_path, file_name)
                save_folder_path = os.path.dirname(os.path.abspath(save_path))
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                cv2.imwrite(save_path, processed_image)
        self.save_dataset_config(processed_dataset_path)
    def convert_image(self, image_path):
        face_area_threshold = self.config['image']['face_area_threshold']

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(image, 2)
        if len(rects) != 1:
            return None
        else:
            area = get_area(rects[0])
            if area < face_area_threshold:
                return None
            aligned_face = align_face(self.config['image'], rects[0], self.face_aligner, image, gray_image)
            return aligned_face

    def save_dataset_config(self, processed_dataset_path):
        with open(os.path.join(processed_dataset_path, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


