import sys, mariadb, os

host = "127.0.0.1"
port = 3306
DbUser = "user1"
DbPassword = "virdis"
Database = "Virdis"

try:
    DbObject = mariadb.connect(
        host=host,
        port=port,
        user=DbUser,
        password=DbPassword,
        database=Database
    )
except Exception as e:
    print("Cant connect")
    # raise ConntionError("Could not connect to the datasee {}".format(e))
    sys.exit(0)
else:
    cursor = DbObject.cursor()
    sql = "select * from Deposit where points is null"
    cursor.execute(sql)
    queryResponse = cursor.fetchall()
    #print("values is " + str(len(queryResponse)))
target_server = "mongodb+srv://codem:1100111@virdismart.nx3p6.mongodb.net/Virdsmart?retryWrites=true&w=majority"

if len(queryResponse) > 0:
    import numpy as np
    import os,pymongo
    import six.moves.urllib as urllib
    import tarfile
    import tensorflow as tf
    import zipfile

    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image

    from utils import label_map_util

    from utils import visualization_utils as vis_util

    from reward import reward

    import cv2, sys

    print("done importing.")

    # This script is more on the outdated side as TensorFlow has updated the object_detection_tutorial.py script.
    # I am currently using this code as the new script has errors in it when trying to run on a local PC I haven't worked
    # out yet. This code has also been edited to work on local PC as it previously did not. Check last few lines

    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
    #
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

    # In[4]:

    # What model to download.
    MODEL_NAME = 'new_graph'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

    NUM_CLASSES = 1

    # ## Download Model

    # In[5]:

    # opener = urllib.request.URLopener()
    # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #    file_name = os.path.basename(file.name)
    #    if 'frozen_inference_graph.pb' in file_name:
    #       tar_file.extract(file, os.getcwd())

    # ## Load a (frozen) Tensorflow model into memory.

    # In[6]:

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    # In[7]:

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # ## Helper code

    # In[8]:

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    # # Detection

    # In[9]:

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    # PATH_TO_TEST_IMAGES_DIR = 'test_images'
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]  # change this value if you want to add more pictures to test

    def detect_bottle(image_path, mass):
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in
                            range(1, 4)]  # change this value if you want to add more pictures to test
        PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)

        # In[10]:

        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                i = 0
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                if image is None:
                    return 0
                (im_width, im_height) = image.size
                image = cv2.imread(image_path)
                # image_np =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                image_np = np.array(image).astype(np.uint8)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                ymin, ymax, xmin, xmax = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                # plt.show()    # when running on local PC, matplotlib is not set for graphical display so instead we
                # can save the outputs to a folder I named outputs (make sure to add this folder into object_detection)
                plt.savefig("outputs/detection_output{}.png".format(i))
                i = i + 1
                bbox = np.squeeze(boxes)

                wid = ymax - ymin
                len = xmax - xmin

                sys.stdout.write(str(wid) + "\n" + str(len))
                data = reward(wid, len, mass)
                #print(data)
                return data


    #detect_bottle("/home/smartbin_2.0/img/img19417.jpg", 7)


# connect online
client = pymongo.MongoClient(target_server)

db = client['Virdismart']

collection_user = db['users']
collection_deposit = db['Deposit']

def parseData(dataBlock,binKeys):
    zipped = dict()
    bunch = list()

    for singleton in dataBlock:
        bunch.append(dict(zip(binKeys,singleton)))

    return bunch

def SyncData(collection_name , dataset):
    try:
        print("updating online database ... ")
        collection_name.insert_many(dataset)

    except Exception as e:
        print("Error updating the online store")
        mailing_content=",\nLogs: "+ "table "+ collection_name + " threw an error"
    else:
        query = "UPDATE Deposit set uploaded=? where uploaded=?"
        curr.execute(
            query,(1,0)
            )
        conn.commit()
        print("Sync successfull")
    finally:
        print("exiting")

def uploadData():
    query = "SELECT binId,userId,massThrown,sizeThrown,fillA,fillB,utype,log,points FROM Deposit WHERE uploaded=? and points is not null"

    try:
        curr.execute(
        query,(0,)
    )
    except Exception as e:
        mailing_content ="\n "+ "Error executint query " + query

    else:
        pending_deposit_upload_list = curr.fetchall()

        if len(pending_deposit_upload_list)>0:
            columns = ("binId","userId","massThrown","sizeThrown","fillA","fillB","utype","log","points")
            parsed = parseData(pending_deposit_upload_list,columns)
            print(parsed)
            print("Duncan the data is: ")

            SyncData(collection_deposit,parsed)

    #SendMail.send_email("cirenenum@gmail.com","gpJ513672","cirenenum@gmail.com","Smart Bin Logs",mailing_content)
    print("Database updated online")

def batchUpdateDb():
    updateEntries = dict()

    query = "SELECT userId as cardId,image,massThrown from Deposit where points is null"
    cursor.execute(query)
    DbQuery = cursor.fetchall()
    #print(DbQuery)
    print("querying")
    query = "update Deposit set points=? where image=?"
    for entry in DbQuery:
        print(os.path.exists(entry[1])," " ,entry[1])
        if os.path.exists(entry[1]):
            print("Rewarding")
            print("The mass is " + str(entry[2]))
            #print(type(entry[1]))
            points = detect_bottle(entry[1], entry[2])
            print(points)
            cursor.execute(query, (points, entry[1]))
            DbObject.commit()

    #save the changes on the database
    DbObject.close()
uploadData()

batchUpdateDb()

# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
#
#
# # In[10]:
#
# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         i = 0   # add variable for a janky fix
#         for image_path in TEST_IMAGE_PATHS:
#             image = Image.open(image_path)
#             # the array based representation of the image will be used later in order to prepare the
#             # result image with boxes and labels on it.
#             image_np = load_image_into_numpy_array(image)
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Each box represents a part of the image where a particular object was detected.
#             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Each score represent how level of confidence for each of the objects.
#             # Score is shown on the result image, together with the class label.
#             scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#             # Actual detection.
#             (boxes, scores, classes, num_detections) = sess.run(
#                 [boxes, scores, classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded})
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=8)
#             plt.figure(figsize=IMAGE_SIZE)
#             plt.imshow(image_np)
#             # plt.show()    # when running on local PC, matplotlib is not set for graphical display so instead we
#             # can save the outputs to a folder I named outputs (make sure to add this folder into object_detection)
#             plt.savefig("outputs/detection_output{}.png".format(i))
#             i = i+1
#             bbox = np.squeeze(boxes)
#
#             ymin = boxes[0][i][0]*12
#             xmin = boxes[0][i][1]*8
#             ymax = boxes[0][i][2]*12
#             xmax = boxes[0][i][3]*8
#
#             wid = ymax - ymin
#             len = xmax - xmin
#
#             sys.stdout.write(str(wid)+"\n")
#
#             reward(wid,len,8)

