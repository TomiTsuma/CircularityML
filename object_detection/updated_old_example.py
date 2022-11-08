import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from tensorflow.keras.preprocessing.image import img_to_array

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO



from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# This script is more on the outdated side as TensorFlow has updated the object_detection_tutorial.py script.
# I am currently using this code as the new script has errors in it when trying to run on a local PC I haven't worked
# out yet. This code has also been edited to work on local PC as it previously did not. Check last few lines




# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


MODEL_NAME = 'new_graph'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, './new_graph/frozen_inference_graph.pb')

PATH_TO_CKPT = filename


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:




# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap("./object_detection/data/mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (299, 299, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'

TEST_IMAGE_PATHS = ["C:/Users/Paylend/Documents/Circularity/object_detection/model/gATbBClmfa.png"]
IMAGE_SIZE = (12, 8)


# In[10]:
def getBottle(image_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)

    i = 0
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")
    image.save("object_detection/outputs/initial.png")


    image_np = np.array(image)
    image = image_np[:, :, ::-1].copy()
    # image = image.resize((299,299), Image.HAMMING)

    # image_np = image.convert("RGB")
    # image_np = img_to_array(image_np)

    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image)
    plt.show() 
    plt.savefig("object_detection/outputs/file2.png")

    image_np_expanded = np.expand_dims(image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})


    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.6)

    plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image)
    # plt.show() 
    plt.savefig("object_detection/outputs/file.png")
    
    i = i+1

