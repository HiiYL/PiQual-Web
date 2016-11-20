## Hack to modify dim ordering
import os

filename = os.path.join(os.path.expanduser('~'), '.keras', 'keras.json')
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    f.write('{"backend": "tensorflow","floatx": "float32","epsilon": 1e-07,"image_dim_ordering": "th"}')


from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from flask import g,jsonify
from werkzeug import secure_filename

import os
import numpy as np
import cv2
from keras.models import load_model


from model import VGG_19_GAP_functional
from utils import preprocess_image,deprocess_image

import netifaces as ni

UPLOAD_FOLDER = 'uploads/'
HEATMAP_FOLDER = 'heatmaps/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER


model = VGG_19_GAP_functional("aesthestic_gap_weights_1_tensorflow.h5", heatmap=True)

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

        original_img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        width, height, _ = original_img.shape

        im = preprocess_image(original_img)

        out = model.predict(im)

        out[1] = out[1][0,:,:,:] 

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = out[1].shape[1:3])


        class_weights = model.layers[-1].get_weights()[0]

        class_to_visualize = 1 # 0 for bad, 1 for good
        for i, w in enumerate(class_weights[:, class_to_visualize]):
                cam += w * out[1][i, :, :]
        print("predictions", out[0])
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        im = heatmap*0.5 + original_img

        cv2.imwrite(os.path.join(app.config['HEATMAP_FOLDER'], filename),im)

        print(out[0].shape)

        return render_template('index.html',object_class = out, filename=filename)
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def api():
    if request.method == 'POST':
      file = request.files['file']
      print(file.filename)
      if file:#and allowed_file(file.filename.lower()):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.seek(0)
        dat = {'score': 5}
        return jsonify(**dat)

      #   im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

      #   im = cv2.resize(im, (224,224)).transpose((2,0,1))
      #   im = np.expand_dims(im,axis=0)

      #   out = model.predict(im)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
  return send_from_directory(app.config['HEATMAP_FOLDER'],
    filename)

if __name__ == "__main__":
    # app.debug = True
    
    # print("YOUR IP ADDRESS IS: {0}".format(ni.ifaddresses('en0')[2][0]['addr']))
    app.run(host='0.0.0.0')
    # app.run()
