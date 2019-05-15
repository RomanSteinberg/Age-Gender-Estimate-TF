import os
import cv2
import dlib
import json
import yaml
from tqdm import tqdm
from age_gender.preprocess.face_aligner import FaceAligner


def get_area(rect):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (bottom - top) * (right - left)


class Converter:

    def __init__(self, config):
        self.config = config
        self.json = config['general']['dataset_path']

        self.shape_predictor = 'shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.face_aligner = FaceAligner(config['image'], self.predictor)

    def convert_dataset(self):
        dataset_folder = os.path.dirname(self.json)
        processed_dataset_path = self.config['general']['processed_dataset_path']
        with open(self.json) as f:
            dataset = json.load(f)

        new_dataset = []
        bad_gender_cnt = 0
        for record in tqdm(dataset, bar_format='Progress {bar} {percentage:3.0f}% [{elapsed}<{remaining}'):
            if not isinstance(record['gender'], float):
                bad_gender_cnt += 1
                continue

            file_name = record['file_name'][0]
            image_path = os.path.join(dataset_folder, file_name)
            save_path = os.path.join(processed_dataset_path, file_name)
            if os.path.exists(save_path):
                continue
            processed_image = self.convert_image(image_path)
            if processed_image is not None:
                save_folder_path = os.path.dirname(os.path.abspath(save_path))
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                cv2.imwrite(save_path, processed_image)
                new_dataset.append({'file_name': file_name, 'gender': int(record['gender']), 'age': record['age']})
        with open(os.path.join(processed_dataset_path, 'dataset.json'), 'w') as f:
            json.dump(new_dataset, f)
        self.save_dataset_config(processed_dataset_path)
        print('Records with incorrect gender: ', bad_gender_cnt)
        print('Total records transformed %d/%d' % (len(new_dataset), len(dataset)))

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
            aligned_face = self.face_aligner.align(image, gray_image, rect)
            return aligned_face

    def save_dataset_config(self, processed_dataset_path):
        with open(os.path.join(processed_dataset_path, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


