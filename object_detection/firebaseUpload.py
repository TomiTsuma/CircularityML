
import firebase_admin
from firebase_admin import credentials, initialize_app, storage
# from predict import predict
# from updated import detect_bottle
# from updated_old_example import getBottle
import requests

config ={
    "apiKey": "AIzaSyByKE87eHZHRTPyEjcJ-a9GKlmlt69ykTA",
    "authDomain": "circularity-space.firebaseapp.com",
    "projectId": "circularity-space",
    "storageBucket": "circularity-space.appspot.com",
    "messagingSenderId": "791489878844",
    "appId": "1:791489878844:web:81ca052cded94553bd5c39",
    "measurementId": "G-L1L8HC08MW"
}


cred = credentials.Certificate("C:/Users/Paylend/Documents/Circularity/object_detection/model/circularityai-firebase-adminsdk-lpnti-5da5a90f34.json")
firebase_admin.initialize_app(cred, {'storageBucket':"circularityai.appspot.com"})


def uploadFile(uri, name):
    path_on_cloud = "images/"+name
    path_local = uri
    bucket = storage.bucket()

    blob = bucket.blob(path_on_cloud)
    blob.upload_from_filename(path_local)

    blob.make_public()
    print(blob.public_url)



    # url = 'https://circularity.eu-gb.cf.appdomain.cloud/post-image'
    url = "http://127.0.0.1:5000/post-image"

    x = requests.post(url, json={"path": blob.public_url})


    print(x.content)
    print(x.json())


uploadFile("C:/Users/Paylend/Documents/Circularity/object_detection/fdIAhgw3kM.png", "image.png")
