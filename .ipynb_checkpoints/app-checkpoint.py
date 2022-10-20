from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model1.h5')

class_dict = {0: 'Daun nangka', 1: 'Daun pepaya'}

labels = ['Bagong', 'Puntadewa', 'Werkudara']

def predict_label(img_path):
    query = cv2.imread(img_path)
    output = query.copy()
    query = cv2.resize(query, (224, 224))
    q = []
    q.append(query)
    q = np.array(q, dtype='float') / 255.0
    prediction = model.predict(query[np.newaxis,...])[0]
    print("Nilai yang diprediksi adalah:",prediction)
    predicted_label = np.argmax(prediction)
    print("Label yang diprediksi adalah:",predicted_label,":",labels[predicted_label])
    
    return labels[predicted_label]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)