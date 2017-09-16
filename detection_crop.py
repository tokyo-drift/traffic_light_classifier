import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

## Some snapshots (saved avia rosrun image_view image_saver image:=/image_color)
## rosbag    https://drive.google.com/file/d/0B44W6q42oyuDVEZuVUNWRURHVXM/
## simulator https://drive.google.com/file/d/0B7dX6v9AKrKxSVVnVG8ybXpYV3M/


dirname = 'simulator_images' if len(sys.argv) < 2 else argv[1]

# Activate optimizations for TF
config = tf.ConfigProto(device_count = {'GPU': 0})
jit_level = tf.OptimizerOptions.ON_1
config.graph_options.optimizer_options.global_jit_level = jit_level

# Function to load a graph from a protobuf file
def load_graph(graph_file):
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
            tf.import_graph_def(gd, name='')
        return sess.graph

detection_graph = load_graph(os.path.join('model', 'model_detection.pb'))
print("Model loaded!")

# Those are the paths for our test images
image_paths = [ os.path.join('test_images', 'image{}.jpg'.format(i)) for i in range(1, 10) ]

# Load the images and store them into a list
images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    images.append(image)

print("Test images loaded!")

# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# This is the class from MS COCO dataset, we only need class 10 = traffic light
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


def extractBoxes(boxes, scores, classes, confidence):
    # Prepare stuff
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    # Get bounding boxes
    boxList = []
    for i in range(min(3, boxes.shape[0])):
        # Check if criteria match
        if scores[i] > confidence and classes[i] == 10:
            # Create a tuple for earch box
            box = tuple(boxes[i].tolist())

            # Extract box corners
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                              ymin * im_height, ymax * im_height)

            # Expand them a little bit
            left = left - 5
            if left < 0:
                left = 0
            top = top - 5
            if top < 0:
                top = 0
            bottom = bottom + 5
            if bottom > im_height:
                bottom = im_height
            right = right + 5
            if right > im_width:
                right = im_width
            box = int(left), int(right), int(top), int(bottom)

            # Add them to the list
            boxList.append(box)

    return boxList

def drawBoxes(boxList, colors, image):
    for box, color in zip(boxList, colors):
            left, right, top, bottom = box
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    return image

if not os.path.exists(dirname + '_crop'): os.mkdir(dirname + '_crop')

# Go through all images
for color in ['red', 'yellow', 'green']:

    from_dir = dirname + os.path.sep + color
    to_dir = dirname + '_crop' + os.path.sep + color

    if not os.path.exists(to_dir): os.mkdir(to_dir)

    for filename in [f for f in os.listdir(from_dir) if os.path.isfile(from_dir + os.path.sep + f)]:

        # Load image and convert
        image = cv2.imread(from_dir + os.path.sep + filename)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        im_height, im_width, _ = image.shape
        image_expanded = np.expand_dims(image, axis=0)

        ## Do traffic lights detection
        with tf.Session(graph=detection_graph, config=config) as sess:
            # Do the inference
            boxes, scores, classes = sess.run(
                [detection_boxes, detection_scores, detection_classes],
                feed_dict={image_tensor: image_expanded})

            # Extract boxes
            boxList = extractBoxes(boxes, scores, classes, 0.5)

            for left, right, top, bottom in boxList:
                sub_image = image[top:bottom, left:right]
                plt.imsave(to_dir + '/' + filename, sub_image)
