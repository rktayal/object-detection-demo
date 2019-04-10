"""
 The script loads the model into memory,
 performs detections on your webcam
"""

import os
import sys
import cv2
import argparse
#import utils as ut
import numpy as np
import tensorflow as tf

#from PIL import Image
#from io import  StringIO
#from collections import defaultdict
#from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from imutil.app_utils import FPS

def load_model(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    session = tf.Session(graph=detection_graph)
    return detection_graph, session


def load_label_map(PATH_TO_LABELS, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


if __name__ == "__main__":
    # Read a video 
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', "--num-frames", type=int, default=100,
            help="# of frames to loop over FPS test")
    ap.add_argument('-d', "--display", type=int, default=-1,
            help="whether or not frame should be displayed")
    args = vars(ap.parse_args())
    cap = cv2.VideoCapture(0)   # change only if you have more than one webcams
    PATH_TO_CKPT = "./model/frozen_inference_graph.pb"
    PATH_TO_LABELS = "./model/mscoco_label_map.pbtxt"
    NUM_CLASSES = 90

    detection_graph, sess = load_model(PATH_TO_CKPT)
    category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)

    with detection_graph.as_default():
        fps = FPS().start()
        while fps._numFrames < args["num_frames"]:
            # Read frame from camera
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract Detection Boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract Detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract Detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual Detection
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict = {image_tensor: image_np_expanded})

            if args["display"] > 0:
                # Visualization of the result of the detection
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                
                # Display output
                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                if cv2.waitKey(25) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
            fps.update()
    fps.stop()
    print ("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
    print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    sess.close()





