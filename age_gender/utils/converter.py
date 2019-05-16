import os
import cv2
import dlib
import json
import yaml
from tqdm import tqdm
from age_gender.preprocess.face_aligner import FaceAligner
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_area(rect):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (bottom - top) * (right - left)


class Converter:

    def __init__(self, config, slice_limits):
        self.config = config
        self.slice = slice_limits
        self.dataset_path = config['general']['dataset_path']

        self.shape_predictor = 'shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.face_aligner = FaceAligner(config['image'], self.predictor)

    def convert_dataset(self):
        dataset_folder = os.path.dirname(self.dataset_path)
        processed_dataset_path = self.config['general']['processed_dataset_path']
        with open(self.dataset_path) as f:
            dataset = json.load(f)[self.slice[0], self.slice[1]]

        new_dataset = []
        bad_gender_cnt = 0
        small_faces_cnt = 0
        for record in tqdm(dataset, bar_format='Progress {bar} {percentage:3.0f}% [{elapsed}<{remaining}'):
            if not isinstance(record['gender'], float):
                bad_gender_cnt += 1
                continue

            file_name = record['file_name'][0]
            image_path = os.path.join(dataset_folder, file_name)
            save_path = os.path.join(processed_dataset_path, file_name)
            if not os.path.exists(save_path):
                processed_image = self.convert_image(image_path)
                if processed_image is None:
                    small_faces_cnt += 1
                    continue
                else:
                    save_folder_path = os.path.dirname(os.path.abspath(save_path))
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)
                    cv2.imwrite(save_path, processed_image)
            new_dataset.append({'file_name': file_name, 'gender': int(record['gender']), 'age': record['age']})
        return new_dataset, bad_gender_cnt, small_faces_cnt

    def convert_image(self, image_path):
        face_area_threshold = self.config['image']['face_area_threshold']

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    @staticmethod
    def run_job(config, slice_limits):
        converter = Converter(config, slice_limits)
        return converter.convert_dataset()


class ConverterManager:
    def __init__(self, config, n_jobs):
        self.n_jobs = n_jobs
        self.config = config
        self.dataset_path = config['general']['dataset_path']
        self.processed_dataset_path = self.config['general']['processed_dataset_path']

    def run(self):
        with open(self.dataset_path) as f:
            dataset_len = len(json.load(f))

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = list()
            subset_len = dataset_len / self.n_jobs
            for start in range(0, dataset_len, subset_len):
                finish = start + subset_len if start + 2*subset_len < dataset_len else dataset_len
                futures.append(executor.submit(Converter.run_job, self.config, (start, finish)))

            new_dataset = list()
            bad_gender_cnt, small_faces_cnt = 0, 0
            for job in as_completed(futures):
                new_dataset += job.result()[0]
                bad_gender_cnt += job.result()[1]
                small_faces_cnt += job.result()[2]

        with open(os.path.join(self.processed_dataset_path, 'dataset.json'), 'w') as f:
            json.dump(new_dataset, f)
        self._save_dataset_config()
        print('Records with incorrect gender: ', bad_gender_cnt)
        print('Records with small faces: ', small_faces_cnt)
        print('Total records transformed %d/%d' % (len(new_dataset), dataset_len))

    def _save_dataset_config(self):
        with open(os.path.join(self.processed_dataset_path, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
