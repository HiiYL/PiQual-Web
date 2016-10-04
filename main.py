from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from flask import g
from werkzeug import secure_filename

import os
from ava import *
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model

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
      if file and allowed_file(file.filename.lower()):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.seek(0)

        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        im = cv2.resize(im, (224,224)).transpose((2,0,1))
        im = np.expand_dims(im,axis=0)

        out = model.predict(im)

        return render_template('index.html',object_class = out, filename=filename)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

if __name__ == "__main__":
    # app.debug = True
    model = load_model('linear_1e5_full_5.h5')
    app.run(host='0.0.0.0')
    # app.run()
