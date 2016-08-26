from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from flask import g
from werkzeug import secure_filename

import os
from ava import *
import pandas as pd
import cv2, numpy as np
from scipy import ndimage, misc

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

        return render_template('response.html',object_class = out, filename=filename)
    return app.send_static_file('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

if __name__ == "__main__":
    # app.debug = True
    model = VGG_19('ava_vgg_19_1.0_5.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    app.run()
    # app.run()
