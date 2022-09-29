import io
import logging
import os
import time

import numpy as np

import tensorflow as tf

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import cv2

CLASSIFIER_FILENAME = '/meccano/meccano_real_en4.tflite'
DETECTOR_NAME = '/meccano/meccano_efficientdet_classes.tflite'
DATASET_PATH = '/meccano/test_data.tfrecord'
THRESHOLD = 0.4

IMAGE_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

def main():
    base_options = core.BaseOptions(
        file_name=CLASSIFIER_FILENAME, use_coral=False, num_threads=8)
    classification_options = processor.ClassificationOptions(
        max_results=1, score_threshold=THRESHOLD)
    options = vision.ImageClassifierOptions(
        base_options=base_options, classification_options=classification_options)

    classifier = vision.ImageClassifier.create_from_options(options)
    
    base_options = core.BaseOptions(
        file_name=DETECTOR_NAME, use_coral=False, num_threads=8)
    detection_options = processor.DetectionOptions(
        max_results=1, score_threshold=THRESHOLD)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    od_good = 0
    od_bad = 0
    
    cl_good = 0
    cl_bad = 0

    dataset = tf.data.TFRecordDataset(DATASET_PATH)
    count = 0
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for value in dataset:
        parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)
        img_bytes = parsed['image/encoded'].numpy()
        np_data = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        if len(detection_result.detections) == 0:
            od_bad += 1
            cl_bad += 1
            continue

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
            od_bad += 1
            cl_bad += 1
            continue

        if (bbox.width < 1) or (bbox.height < 1):
            od_bad += 1
            cl_bad += 1
            continue

        correct_label = parsed['image/object/class/text'].values[0].numpy().decode('utf8')        
        category_name = category.category_name
        if category_name == correct_label:
            od_good += 1
        else:
            od_bad += 1

        # Converting to JPEG and back again raises performance
        bgr_crop = bgr_image[top:bottom, left:right]
        _, jpeg_crop = cv2.imencode(".jpg", bgr_crop)
        bgr_crop = cv2.imdecode(jpeg_crop, cv2.IMREAD_COLOR)
        rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        tensor_image = vision.TensorImage.create_from_array(rgb_crop)
        categories = classifier.classify(tensor_image)

        if len(categories.classifications[0].categories) == 0:
            cl_bad += 1
            continue

        assert len(categories.classifications[0].categories) == 1
        category = categories.classifications[0].categories[0]
        pred_label_name = category.category_name
                    
        if pred_label_name == correct_label:
            cl_good += 1
        else:
            cl_bad += 1
            # print(pred_label_name, category_name, correct_label)

        count += 1
        if count == 200:
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            print('time for 200 in ns', (end - start))
            print('od good:', od_good)
            print('od bad:', od_bad)

            print('cl good:', cl_good)
            print('cl bad:', cl_bad)
            count = 0
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    print('od good:', od_good)
    print('od bad:', od_bad)

    print('cl good:', cl_good)
    print('cl bad:', cl_bad)
    

if __name__ == '__main__':
    main()
            
