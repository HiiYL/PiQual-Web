from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from werkzeug import secure_filename

import os
from vgg import *
import pandas as pd
import cv2, numpy as np

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
      names = pd.read_table("synset_words.txt", header = None,names=["names"])
      file = request.files['file']
      if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.seek(0)

        raw_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
        im = cv2.resize(raw_image, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)

        model = VGG_16('vgg16_weights.h5')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict(im)
        object_class = str(names.loc[np.argmax(out)]["names"])[10:]
        return render_template('response.html',object_class = object_class, filename=filename)
    return app.send_static_file('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

if __name__ == "__main__":
    app.debug = True
    app.run()