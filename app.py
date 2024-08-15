from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
try:
    model = pickle.load(open('heart_attacl_model.pkl', 'rb'))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

app = Flask(__name__)

@app.route('/')
def home():
    logging.debug("Home endpoint accessed.")
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = request.json['age']
        sex = request.json['sex']
        cp = request.json['cp']
        trtbps = request.json['trtbps']
        chol = request.json['chol']
        fbs = request.json['fbs']
        restecg = request.json['restecg']
        thalachh = request.json['thalachh']
        exng = request.json['exng']
        oldpeak = request.json['oldpeak']
        slp = request.json['slp']
        caa = request.json['caa']
        thall = request.json['thall']

        logging.debug(f"Received input: {request.json}")

        # Convert oldpeak to float in case it's provided as a string
        oldpeak = float(oldpeak)
        
        # Prepare the input data for the model
        input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        
        # Make the prediction
        result = model.predict(input_data)[0]
        logging.info(f"Prediction result: {result}")

        # Return prediction result
        print("ye mai hu",result)
        if result == 1:
            return jsonify({'result': 'You have a heart attack'})
        else:
            return jsonify({'result': 'You do not have a heart attack'})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
