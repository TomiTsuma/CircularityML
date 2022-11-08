import firebase_admin
from firebase_admin import credentials, initialize_app, storage
import requests

from google.cloud import storage
from google.oauth2 import service_account

config ={
    "apiKey": "AIzaSyByKE87eHZHRTPyEjcJ-a9GKlmlt69ykTA",
    "authDomain": "circularity-space.firebaseapp.com",
    "projectId": "circularity-space",
    "storageBucket": "circularity-space.appspot.com",
    "messagingSenderId": "791489878844",
    "appId": "1:791489878844:web:81ca052cded94553bd5c39",
    "measurementId": "G-L1L8HC08MW"
}

cred = credentials.Certificate("/Users/georgreen/admin/circularity-space-firebase-adminsdk-thko5-07a8ad9e71.json")
firebase_admin.initialize_app(cred, {'storageBucket':"circularity-space.appspot.com"})

#credentials = service_account.Credentials.from_service_account_file("/Users/georgreen/admin/circularity-space-firebase-adminsdk-thko5-07a8ad9e71.json")
#storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob('images/0YB21Zi8pU.png').download_to_filename('image_1.png')

#Access the folder in firebase storage with images
#Get a list of all the images in the folder
#Loop through the images in the list while downloading them to the laptop


#Downloading a single image from firebase storage
def download_blob():
    credentials = service_account.Credentials.from_service_account_file("/Users/georgreen/admin/circularity-space-firebase-adminsdk-thko5-07a8ad9e71.json")
    storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob('images/0YB21Zi8pU.png').download_to_filename('image_1.png')

download_blob()

#Uploading a single image from firebase storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    credentials = service_account.Credentials.from_service_account_file("/Users/georgreen/admin/circularity-space-firebase-adminsdk-thko5-07a8ad9e71.json")
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

upload_blob(firebase_admin.storage.bucket().name, 'photo.jpg', 'model_images/bottle.jpg')
