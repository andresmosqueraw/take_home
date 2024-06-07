import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Usar la variable de entorno MODEL_PATH para definir la ruta del modelo
model_path = os.environ.get('MODEL_PATH', './mle-intv-main/model.pkl')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)  # Convertir los datos a un DataFrame
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return jsonify({'predictions': predictions.tolist(), 'probabilities': probabilities.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
