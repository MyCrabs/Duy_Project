from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
model = load_model('MyModel.h5')  

class_names = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower','chilli pepper',
               'corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango',
               'onion','orange','paprika','pear','peas','pineapple','pomegranate','potato','raddish','soy_beans',
               'spinach','sweetcorn','sweetpotato','tomato','turnip','watermelon']

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route('/')
def home():
    return render_template('upload.html')

    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        img_array = load_and_preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        
        return render_template('result.html', filepath=f'uploads/{filename}', predicted_class=predicted_class)
    else:
        return redirect(request.url)

@app.route('/upload_url', methods=['POST'])
def upload_file_url():
    url = request.form['url']
    if url == '':
        return redirect(request.url)
  
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(upload_folder, filename)
        img.save(filepath)
        
        img_array = load_and_preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        
        return render_template('result.html', filepath=f'uploads/{filename}', predicted_class=predicted_class)
    except Exception as e:
        return str(e)
    

if __name__ == "__main__":
    app.run(debug=True)
