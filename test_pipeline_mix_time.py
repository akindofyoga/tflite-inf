import io
import os
import time

import tensorflow as tf

from PIL import Image

import mpncov
from torchvision import transforms
import torch

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import cv2

CLASSIFIER_FILENAME = '/stirling_voc/model_best.pth.tar'
DETECTOR_NAME = '/stirling_voc/sitrling_all_classes.tflite'
TEST_IMAGES = '/stirling_voc/test_images'
THRESHOLD = 0.4
LABELS = ['1screw', '2screws', '3screws', 'nocylinder', 'nopad', 'nopiston', 'noring', 'noscrews']


class Pipeline:
    def __init__(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                normalize,
        ])

        classifier_representation = {
            'function': mpncov.MPNCOV,
            'iterNum': 5,
            'is_sqrt': True,
            'is_vec': True,
            'input_dim': 2048,
            'dimension_reduction': None,
        }

        freezed_layer = 0
        self.model = mpncov.Newmodel(classifier_representation,
                                     len(LABELS), freezed_layer)
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.cuda()
        trained_model = torch.load(CLASSIFIER_FILENAME)
        self.model.load_state_dict(trained_model['state_dict'])
        self.model.eval()    

        base_options = core.BaseOptions(
            file_name=DETECTOR_NAME, use_coral=False, num_threads=8)
        detection_options = processor.DetectionOptions(
            max_results=1, score_threshold=THRESHOLD)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def inf(self, img_full_path, correct_label):
        bgr_image = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
        pil_img = Image.open(img_full_path)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = self.detector.detect(input_tensor)

        if len(detection_result.detections) == 0:
            return False

        assert len(detection_result.detections) == 1
        detection = detection_result.detections[0]
        category = detection.categories[0]
        assert category.score > THRESHOLD

        bbox = detection.bounding_box
        left = bbox.origin_x
        right = bbox.origin_x + bbox.width
        top = bbox.origin_y
        bottom = bbox.origin_y + bbox.height

        if (top < 0) or (left < 0):
            return False

        if (bbox.width < 1) or (bbox.height < 1):
            return False

        # Converting to JPEG and back again raises performance
        pil_img = pil_img.crop((left, top, right, bottom))
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, 'jpeg')

        pil_img = Image.open(img_bytes)
        transformed = self.transform(pil_img).cuda()

        output = self.model(transformed[None, ...])
        _, pred = output.topk(1, 1, True, True)
        classId = pred.t()

        pred_label_name = LABELS[classId]

        return (pred_label_name == correct_label)
        

def main():
    count = 0
    good = 0
    bad = 0

    pipeline = Pipeline()
    
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for correct_label in os.listdir(TEST_IMAGES):
        img_dir = os.path.join(TEST_IMAGES, correct_label)
        
        for img_name in os.listdir(img_dir):
            img_full_path = os.path.join(img_dir, img_name)

            if pipeline.inf(img_full_path, correct_label):
                good += 1
            else:
                bad += 1

            if count == 200:
                end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                print('time for 200 in ns', (end - start))
                print('good:', good)
                print('bad:', bad)
                
                count = 0
                start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            count += 1

    print('good:', good)
    print('bad:', bad)

    
if __name__ == '__main__':
    main()
            
