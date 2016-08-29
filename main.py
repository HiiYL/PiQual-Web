from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from flask import g
from werkzeug import secure_filename

import os
from ava import *
import pandas as pd
import numpy as np
from scipy import ndimage, misc
from keras.models import load_model

import netifaces as ni

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
      file = request.files['file']
      if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.seek(0)

        image = ndimage.imread(file, mode="RGB")
        image_resized = misc.imresize(image, (224, 224)).T
        im = np.expand_dims(image_resized,axis=0)

        out = model.predict(im)

        return render_template('index.html',object_class = out, filename=filename)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

if __name__ == "__main__":
    # app.debug = True
    print("YOUR IP ADDRESS IS: {0}".format(ni.ifaddresses('en0')[2][0]['addr']))
    model = load_model('ava_vgg_19_1.0_5.h5')

    
    app.run(host='0.0.0.0')
    # app.run()
