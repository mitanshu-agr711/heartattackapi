from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('heart_attacl_model.pkl', 'rb'))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Ensure model is set to None if loading fails

@app.route('/')
def home():
    logging.debug("Home endpoint accessed.")
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logging.error("Model not loaded. Cannot perform prediction.")
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        print('request strted ')
        # Parse JSON input
        data = request.json
        logging.debug(f"Received input: {data}")

        # Extract features
        age = data['age']
        sex = data['sex']
        cp = data['cp']
        trtbps = data['trtbps']
        chol = data['chol']
        fbs = data['fbs']
        restecg = data['restecg']
        thalachh = data['thalachh']
        exng = data['exng']
        oldpeak = float(data['oldpeak'])  # Ensure oldpeak is a float
        slp = data['slp']
        caa = data['caa']
        thall = data['thall']
        
        # Prepare the input data for the model
        input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        logging.debug(f"Prepared input for model: {input_data}")
        print(input_data)
        # Make the prediction
        result = model.predict(input_data)[0]
        logging.info(f"Prediction result: {result}")

        # Return prediction result
        print('jfuygytuuffh',result)
        if result == 1:
            return jsonify({'result': 'You have a heart attack'})
        else:
            return jsonify({'result': 'You do not have a heart attack'})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
  