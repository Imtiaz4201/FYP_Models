import os
import sys
import cv2

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import json
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import visualization_utils
from object_detection.builders import model_builder


class InferVideo(object):
    def __init__(self) -> None:
        self.video_path = os.path.join(os.getcwd(), "testVideo.mp4")

    def resize_image(self, image, target_size, interP):
        resized_img = cv2.resize(
            image, target_size, interpolation=interP)
        return resized_img

    def convert_png(self, frame):
        encoded_img = cv2.imencode('.png', frame)[1]
        return encoded_img

    def process(self, fm):
        model_path = "C:\\Users\\Imtiaz\\Downloads\\DL_models\\saved_models\\en\\saved_model"
        loaded_model = tf.saved_model.load(model_path)
        category_index = label_map_util.create_category_index_from_labelmap("C:\\Users\\Imtiaz\\Downloads\\DL_models\\annotations\\labelmap.pbtxt",
                                                                            use_display_name=True)
        try:
            # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image_np = np.array(fm)
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = loaded_model(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)
            image_np_with_detections = image_np.copy()
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.3,
                agnostic_mode=False)
            return image_np_with_detections
        except Exception as e:
            print(e)

    def infer(self):
        if os.path.exists(self.video_path):
            cam = cv2.VideoCapture(self.video_path)
            while cam.isOpened():
                ret, frame = cam.read()
                if ret:
                    pm = self.resize_image(frame, (800, 1333), cv2.INTER_CUBIC)
                    # pmp = self.convert_png(pm)
                    # pmp_img = cv2.imdecode(pmp, cv2.IMREAD_UNCHANGED)
                    process_img = self.process(pm)
                    res = self.resize_image(
                        process_img, (400, 400), cv2.INTER_AREA)
                    cv2.imshow("Video", res)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            cam.release()
            cv2.destroyAllWindows()
        else:
            print("video not found!")


if __name__ == "__main__":
    infer_ = InferVideo()
    infer_.infer()
