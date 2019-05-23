from styx_msgs.msg import TrafficLight
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import os
import rospy

from PIL import Image

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TLSimClassifier(object):
    def __init__(self):
        #TODO load classifier
        model_file = DIR_PATH + '/models/sim-classifier-8.h5'
        self.model = load_model(DIR_PATH + '/models/sim-classifier-8.h5')        
        rospy.loginfo("TLSimClassifier load model %s" , model_file)
        
        self.model._make_predict_function() 
        self.graph = tf.get_default_graph()
        self.light_state = TrafficLight.UNKNOWN


    def get_state_string(self, state):
        if (state == 0):
            state_s = "RED"
        elif (state == 1):
            state_s = "YELLOW"
        elif (state == 2):
            state_s = "GREEN"
        else:
            state_s = "UNKNOWN"

        return state_s

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        classification_tl_key = {0: TrafficLight.RED, 1: TrafficLight.YELLOW, 2: TrafficLight.GREEN, 4: TrafficLight.UNKNOWN}

        resized = cv2.resize(image, (80,60))/255.

        test_img = np.array([resized])
        # run the prediction
        with self.graph.as_default():
            model_predict = self.model.predict(test_img)
            if model_predict[0][np.argmax(model_predict[0])] > 0.5:
                self.light_state = classification_tl_key[np.argmax(model_predict[0])]

        return self.light_state

    def transfer_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)


# Simple test
def _main(_):
    flags = tf.app.flags.FLAGS
    image_path = flags.input_image
    model_path = flags.model_path
    simulator = flags.simulator
    print 'image_path=', image_path
    print 'model_path=', model_path
    print 'simulator=', simulator

    image = Image.open(image_path)
    image.show()
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.

    classifier = TLSimClassifier()
    image_np = classifier.transfer_image_into_numpy_array(image)

    classification = classifier.get_classification(image_np)
    state_s = classifier.get_state_string(classification)
    print(state_s)

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input_image', 'light2.jpeg', 'Path to input image')
    flags.DEFINE_string('model_path', 'models/ssd_mobilenet_v1_coco_2018_01_28', 'Path to the second model')
    flags.DEFINE_bool('simulator', False, 'Whether image is coming from a simulator or not')

    tf.app.run(main=_main)
        
