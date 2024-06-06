from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('./mle-intv-main/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Realizar predicciones
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Preparar la respuesta
        response = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
