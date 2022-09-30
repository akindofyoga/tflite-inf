import logging
import os
import time

import numpy as np

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import cv2

CLASSIFIER_FILENAME = '/stirling_voc/r50.tflite'
DETECTOR_NAME = '/stirling_voc/sitrling_all_classes.tflite'
TEST_IMAGES = '/stirling_voc/test_images'
THRESHOLD = 0.4


class Pipeline:
    def __init__(self):
        base_options = core.BaseOptions(
            file_name=CLASSIFIER_FILENAME, use_coral=False, num_threads=8)
        classification_options = processor.ClassificationOptions(
            max_results=1, score_threshold=THRESHOLD)    
        options = vision.ImageClassifierOptions(
            base_options=base_options, classification_options=classification_options)

        self.classifier = vision.ImageClassifier.create_from_options(options)

        base_options = core.BaseOptions(
            file_name=DETECTOR_NAME, use_coral=False, num_threads=8)
        detection_options = processor.DetectionOptions(
            max_results=1, score_threshold=THRESHOLD)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def inf(self, img_path, correct_label):
        bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

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
        bgr_crop = bgr_image[top:bottom, left:right]

        if (top < 0) or (left < 0):
            return False

        if (bbox.width < 1) or (bbox.height < 1):
            return False

        # Converting to JPEG and back again raises performance
        _, jpeg_crop = cv2.imencode(".jpg", bgr_crop)
        bgr_crop = cv2.imdecode(jpeg_crop, cv2.IMREAD_COLOR)
        rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        tensor_image = vision.TensorImage.create_from_array(rgb_crop)
        categories = self.classifier.classify(tensor_image)

        if len(categories.classifications[0].categories) == 0:
            return False

        assert len(categories.classifications[0].categories) == 1
        category = categories.classifications[0].categories[0]
        pred_label_name = category.category_name                

        return (pred_label_name == correct_label)

def main():
    good = 0
    bad = 0
    count = 0

    pipeline = Pipeline()
    
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for correct_label in os.listdir(TEST_IMAGES):
        img_dir = os.path.join(TEST_IMAGES, correct_label)
        
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)

            if pipeline.inf(img_path, correct_label):
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
            
