import os
import cv2
import dlib
import json
import yaml
import numpy as np
from time import monotonic as now
from datetime import timedelta

from age_gender.preprocess.face_aligner import FaceAligner
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_area(rect):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (bottom - top) * (right - left)


class Converter:

    def __init__(self, config, slice_limits, pid):
        self.config = config
        self.slice = slice_limits
        self.pid = pid
        self.dataset_path = config['general']['dataset_path']
        self.shape_predictor = 'shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor)
        self.face_aligner = FaceAligner(config['image'], self.predictor)

    def convert_dataset(self):
        dataset_folder = os.path.dirname(self.dataset_path)
        processed_dataset_path = self.config['general']['processed_dataset_path']
        with open(self.dataset_path) as f:
            dataset = json.load(f)[self.slice[0]: self.slice[1]]

        total = self.slice[1] - self.slice[0]
        new_dataset = []
        bad_gender_cnt = 0
        small_faces_cnt = 0
        start_time = now()
        for ind, record in enumerate(dataset):
            if self.need_print(ind):
                ratio = ind / total
                eta = timedelta(seconds=(now() - start_time) * (1 - ratio) / (ratio + 1e-9))
                print(f'pid: {self.pid}, progress: {round(ratio*100, 1)}% {ind}/{total} images, eta={eta}')
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
        print(f'pid: {self.pid}, total: {total} images, time={now() - start_time}')
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
    def need_print(ind):
        if ind == 0:
            return False
        elif ind == 10:
            return True
        elif ind < 1000:
            return ind % 100 == 0
        else:
            return ind % 1000 == 0

    @staticmethod
    def run_job(config, slice_limits, pid):
        converter = Converter(config, slice_limits, pid)
        return converter.convert_dataset()


class ConverterManager:
    def __init__(self, config):
        self.config = config
        self.n_jobs = config['general']['n_jobs']
        self.dataset_path = config['general']['dataset_path']
        self.processed_dataset_path = self.config['general']['processed_dataset_path']

    def run(self):
        with open(self.dataset_path) as f:
            dataset_len = len(json.load(f))
        dataset_len = 1640
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = list()
            subsets = np.linspace(0, dataset_len, self.n_jobs + 1, dtype=np.int)
            for ind in range(self.n_jobs):
                start = subsets[ind]
                finish = subsets[ind+1]
                print(f'pid: {ind}, slice: [{start}:{finish}]')
                futures.append(executor.submit(Converter.run_job, self.config, (start, finish), ind))

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
