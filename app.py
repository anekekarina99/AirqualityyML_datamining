import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# Inisialisasi aplikasi Flask
with open('model.pkl', 'rb') as file:  
    model = pickle.load(file)


app = Flask(__name__) 

@app.route('/') # Halaman utama
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan nilai dari formulir
    pm10 = request.form['pm10']
    pm25 = request.form['pm25']
    prediction = model.predict(np.array([ int(pm10), int(pm25)]) )# Melakukan prediksi

    output = int(prediction[0])

    return render_template('index.html', prediction_text='Kualitas udara diprediksi: {}'.format(output)) # Merender hasil prediksi


if __name__ == "__main__":
    app.run(debug=True)
