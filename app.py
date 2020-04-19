#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
# from keras import backend as K
# from tensorflow.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import mobilenet
# from tensorflow.keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, request

app = Flask(__name__)

def prepare_image(file):
    file_path = 'images/'
    file.save(file_path + file.filename)
    img = image.load_img(file_path + file.filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    print(keras.applications.mobilenet.preprocess_input(img_array_expanded_dims))
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def return_predictions(file):
    preprocessed_image = prepare_image(file)
#     with CustomObjectScope({'relu6': mobilenet.relu6}):
    model = load_model("models/model.h5")
    print(model.summary)
    predictions = model.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)

    return results

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['POST','GET'])
def upload_file_1():
    if request.method == 'POST': 
        file = request.files['file']
        r = return_predictions(file)
        return ' '.join(map(str, r)) + ' file uploaded successfully'

# if __name__ == '__main__':
#     app.run(debug = False)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)
# In[ ]:




