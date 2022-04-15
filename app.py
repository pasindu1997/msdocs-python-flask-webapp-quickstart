from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
# from fileinput import filename

#Keras
import cv2
from tensorflow.keras.models import load_model

import urllib

#numpy
import numpy as np

import h5py
from azure.storage.blob import BlobClient
from io import BytesIO

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/home',methods=['GET'])
def home():
    return 'GotaGoHome';


@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

con_str = 'DefaultEndpointsProtocol=https;AccountName=stcannellaassist;AccountKey=7LtfWubpLrxVYVc5UijAVshcMQNSxJnfbLthIBhs8uoFteYn1tue2ErTC4K1COXF2T2z4UFa/p1cUFf/MBPKnA==;EndpointSuffix=core.windows.net'
blob_client = BlobClient.from_connection_string(con_str, 'trainedmodel', 'model.h5')
downloader = blob_client.download_blob(0)

# Load ML model
with BytesIO() as f:
    downloader.readinto(f)
    with h5py.File(f, 'r') as h5file:
        model = load_model(h5file)
        model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=["accuracy"])
       
#image Resizing 
def image_resize(inputImage):
    img=cv2.resize(inputImage,(96,96))
    img=np.expand_dims(img,axis=0)
    img=img.astype('float32')
    img=img/255
    return img

# endpoint to get the prediction 
@app.route('/flask/predict',methods=['GET'])
def predictor():
    fileName = request.args.get('filename')
    image_path = "https://stcannellaassist.blob.core.windows.net/cinnamonleadimg/" + fileName
    
    req = urllib.request.urlopen(image_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    loadImage = cv2.imdecode(arr, -1)
    resizedImage =image_resize(loadImage)
    probabilities = model.predict(resizedImage)
    image_class = np.argmax(probabilities,axis=1)
    if image_class == 0:
        return jsonify({'Prediction':'Immatured', 'Probability':(round(float(probabilities[0][0]*100),2))})
    else:
        return jsonify({'Prediction': 'Matured', 'Probability': (round(float(probabilities[0][1]*100),2))})

if __name__ == '__main__':
   app.run()