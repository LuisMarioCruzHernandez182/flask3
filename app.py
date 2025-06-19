from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configura logging
logging.basicConfig(level=logging.INFO)

# Carga el modelo
model = joblib.load("modelo.pkl")

@app.route('/')
def home():
    return render_template("formulario.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Asegúrate de que los nombres coincidan con el formulario
        abdomen = float(request.form.get('abdomen'))
        antena = float(request.form.get('antena'))
        
        # Crea DataFrame con los nombres EXACTOS que usaste al entrenar
        data_df = pd.DataFrame([[abdomen, antena]], 
                             columns=['abdomen', 'antena'])  # Verifica estos nombres
        
        # Realiza predicción
        prediction = model.predict(data_df)
        
        return render_template("formulario.html", 
                            resultado=f"El insecto es: {prediction[0]}")
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return render_template("formulario.html", 
                            resultado=f"Error: {str(e)}"), 400

if __name__ == '__main__':
    app.run(debug=True)