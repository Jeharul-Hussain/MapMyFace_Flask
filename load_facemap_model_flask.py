from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from os.path import join
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
import extract_face_features
import shutil
import cv2
import imageio
import glob

ALLOWED_EXTENSION  =set(['png','jpg','jpeg'])
IMG_SHAPE =50

#os.chdir(r'/home/jeharul/practice_space/data/just_patches/temp')

def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select an image')
    
    if file and allowed_file(file.filename):
        filename = file.filename
        print("File Name --> "+filename)
        split_filename = filename.split(".")[0]
        split_filename = split_filename.split("_")[0]
        print("Image prefix value --> "+split_filename)
        
        model = load_model(os.path.join(os.getcwd()+"/","acne_noacne_classification_extracted.h5"))
        output_path = os.getcwd() + "/test"
        input_path = os.getcwd() + "/input"

        if os.path.exists(output_path) and os.path.isdir(output_path):
            shutil.rmtree(output_path)
            print('test directory removed')

        ImageFile.LOAD_TRUNCATED_IMAGES = False     
        img = Image.open(BytesIO(file.read()))
        #img.load()
        
        #x  = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        imcv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        
        if not os.path.exists(input_path):
            os.makedirs(input_path)
            #
            print("bbbbbbbbbbbbbbbbbbbbbbbb")
            imageio.imwrite(input_path + "/" + filename, imcv)
            print("aaaaaaaaaaaaaaaaaaaaaaaaa")

        #infile = join(in_path, filename)
        
        extract_face_features.extract_patches(input_path + "/" + filename)


        if os.path.exists(input_path) and os.path.isdir(input_path):
            shutil.rmtree(input_path)
            print('input directory removed') 

        if os.path.exists(output_path):
            path = os.path.join(output_path, '*g')

        x = []
        y = []
        label = []
        items = []
        one = 1
        
        for img in glob.glob(path):
            image = cv2.imread(img)
            im_resize = cv2.resize(image,(IMG_SHAPE,IMG_SHAPE), interpolation=cv2.INTER_CUBIC)
            im_resize_copy = im_resize.copy()/255.

            prediction = model.predict(im_resize_copy[np.newaxis,...])
            print("prediction is :{}".format(np.argmax(prediction)))
            label.append(np.argmax(prediction))

        if one in label:
            print("Acne Detected")
            items.append("Acne")
        else:
            print("No Acne Detected")
            items.append("No Acne")

        response = {'pred': items}
        return render_template('ImageML.html', prediction = 'Looks like {}'.format(response))
    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)