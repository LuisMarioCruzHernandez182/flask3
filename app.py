from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar el modelo entrenado
model = joblib.load("model.pkl")
app.logger.info("Modelo cargado correctamente.")

@app.route('/')
def home():
    return render_template("formulario.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Obtener datos del formulario
        abdomen = float(request.form['abdomen'])
        antenna = float(request.form['antenna'])
        
        # Crear DataFrame con los datos
        data_df = pd.DataFrame([[abdomen, antenna]], columns=['abdomen', 'antenna'])
        app.logger.info(f"Datos recibidos: {data_df}")
        
        # Realizar predicción
        prediction = model.predict(data_df)
        app.logger.info(f"Predicción: {prediction[0]}")
        
        # Devolver resultado como JSON
        return jsonify({'category': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)