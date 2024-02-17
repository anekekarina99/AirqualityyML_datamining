import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) # Inisialisasi aplikasi Flask
model = pickle.load(open('model.pkl', 'rb')) # Memuat model yang telah dilatih

@app.route('/') # Halaman utama
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan nilai dari formulir
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # Melakukan prediksi

    output = int(prediction[0])

    return render_template('index.html', prediction_text='Kualitas udara diprediksi: {}'.format(output)) # Merender hasil prediksi


if __name__ == "__main__":
    app.run(debug=True)
