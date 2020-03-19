from styx_msgs.msg import TrafficLight
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

ckpt_path = '../../../classifier/model/frozen_inference_graph_faster_rcnn_v2.pb'

class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph, config=config)
        
    def load_image_into_numpy_array(self, image):
        return np.array(image.getdata()).reshape(image.shape).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = TrafficLight.UNKNOWN
        
        image_np = image
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (scores, classes) = self.sess.run([self.detection_scores, self.detection_classes],feed_dict={self.image_tensor: image_np_expanded})

        if np.max(scores) > 0.6:
            max_score_index = np.argmax(scores)
            light = np.squeeze(classes)[max_score_index]

        if light == TrafficLight.UNKNOWN:
            print("Traffic light state: UNKNOWN")
        elif light == TrafficLight.GREEN:
            print("Traffic light state: GREEN | " + str(np.squeeze(scores)[max_score_index]))
        elif light == TrafficLight.RED:
            print("Traffic light state: RED | " + str(np.squeeze(scores)[max_score_index]))
        elif light == TrafficLight.YELLOW:
            print("Traffic light state: YELLOW | " + str(np.squeeze(scores)[max_score_index]))

        return light
