from flask import Flask
from flask import Flask, request, jsonify
import os

from pathlib import Path

import sys

sys.path
sys.path.append('./object_detection')

from dotenv import load_dotenv

# from object_detection.updated import detect_bottle
from object_detection.updated_old_example import getBottle
from object_detection.model.predict import predict
from database.add_points import addPoints




env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path)

my_env_var = os.getenv('.env')

app = Flask(__name__)
# api = Api(app)

@app.route('/')
def hello_world():
    return 'This is my first API call!'

@app.route('/detect-image', methods=["POST"])
def testpost():
    image = request.get_json()['path']
    user_id = request.get_json()['user_id']
    dictToReturn = {'path':image}
    i = predict(dictToReturn['path'])
    print("This is the prediction", i)
    if(int(i[0][0]) == 1):
        getBottle(dictToReturn['path'])
        addPoints(user_id, 1)
        return jsonify("Bottle Present")
    else:
        return jsonify("No Bottle Present")

    # return jsonify(i[0][0])
