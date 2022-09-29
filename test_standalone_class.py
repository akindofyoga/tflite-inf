import os
import time

from PIL import Image

import mpncov
from torchvision import transforms
import torch

CLASSIFIER_FILENAME = '/stirling/model_best.pth.tar'
TEST_IMAGES = '/stirling/test_crop'
THRESHOLD = 0.4
LABELS = ['1screw', '2screws', '3screws', 'nocylinder', 'nopad', 'nopiston', 'noring', 'noscrews']


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
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
    model = mpncov.Newmodel(classifier_representation,
                            len(LABELS), freezed_layer)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    trained_model = torch.load(CLASSIFIER_FILENAME)
    model.load_state_dict(trained_model['state_dict'])
    model.eval()    
    
    cl_good = 0
    cl_bad = 0

    count = 0
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for correct_label in os.listdir(TEST_IMAGES):
        img_dir = os.path.join(TEST_IMAGES, correct_label)
        
        for img_name in os.listdir(img_dir):
            img_full_path = os.path.join(img_dir, img_name)
            pil_img = Image.open(img_full_path)
            transformed = transform(pil_img).cuda()

            output = model(transformed[None, ...])
            _, pred = output.topk(1, 1, True, True)
            classId = pred.t()

            pred_label_name = LABELS[classId]

            if pred_label_name == correct_label:
                cl_good += 1
            else:
                cl_bad += 1

            count += 1
            if count == 200:
                end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                print('time for 200 in ns', (end - start))

                print('cl good:', cl_good)
                print('cl bad:', cl_bad)
                count = 0
                start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    print('cl good:', cl_good)
    print('cl bad:', cl_bad)
    

if __name__ == '__main__':
    main()
            
