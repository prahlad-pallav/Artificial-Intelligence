# VGG16 -> outline architecture of SSD convolution Neural Network
# VGG16 -> slow training procedure and weights are quite large

# MobileNet is 23x times smaller as compared to VGG16 -> but have same accuracy

import cv2
import numpy as np

# we're not going to bother those whose probability is lesser than 50%
THRESHOLD = 0.6
# the lower the value of suppression threshold: fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.2
SSD_INPUT_SIZE = 320

def construct_class_names(file_name ="class_names"):
    with open(file_name, 'rt') as file:
        names = file.read().rstrip('\n').split('\n')

    return names

def show_detected_objects(img, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        box = all_bounding_boxes[index]
        print(box)
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, object_names[class_ids[index] - 1].upper(), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)


class_names = construct_class_names()
# print(class_names)
# print(type(class_names))

capture = cv2.VideoCapture('objects.mp4')

neural_network = cv2.dnn.DetectionModel('ssd_weights.pb', 'ssd_mobilenet_coco_cfg.pbtxt')
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
neural_network.setInputScale(1.0/127.5)
neural_network.setInputMean((127.5, 127.5, 127.5))
neural_network.setInputSwapRB(True)



while True:

    is_grabbed, frame = capture.read()

    if not is_grabbed:
        break

    class_label_ids, confidences, bbox = neural_network.detect(frame)
    # print(class_label_ids)
    # print(confidences)
    # print(bbox)
    bbox = list(bbox)
    # confidences = confidences[0].tolist()
    confidences = list(confidences)
    # print(confidences)
    # print(type(confidences))

    # indexes of the bounding boxes, we need to keep
    box_to_keep = cv2.dnn.NMSBoxes(bbox, confidences, THRESHOLD, SUPPRESSION_THRESHOLD)
    # print(box_to_keep)
    show_detected_objects(frame, box_to_keep, bbox, class_names, class_label_ids)

    cv2.imshow('SSD Algorithm', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()

