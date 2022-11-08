import cv2,os
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from urllib.request import urlopen
import os


dirname = os.path.dirname(__file__)

print(dirname)

filename = os.path.join(dirname, '/model/model')
print(filename)
print(os.path.exists(dirname))
model = load_model(dirname+'/model/model')


def predict(fileUrl):

    # req = urlopen(fileUrl)
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # img = cv2.imdecode(arr, -1) # 'Load it as it is'
    # img = cv2.resize(img,(299,299))

    response = requests.get(fileUrl)
    img = Image.open(BytesIO(response.content))
    img = img.resize((299,299), Image.HAMMING)
    img = img.convert('RGB')
    print(img.mode)
    img = img_to_array(img)
    print(img.shape)

    img = img/255


    pretrain = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
    bottle_neck_features_predict = pretrain.predict(np.array([img]))[0]

    np.savez('inception_features_prediction', features=bottle_neck_features_predict)
    prediction_data = np.load('inception_features_prediction.npz')['features']
    q = model.predict( np.array( [prediction_data,] )  )
    prediction = q[0]
    prediction = int(prediction)
    print(prediction)
    return(q)

