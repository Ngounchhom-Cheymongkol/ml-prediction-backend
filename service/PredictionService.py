from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def predict_churn(features):
    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([features])
        # One-hot encode geography
        geo_encoded = onehot_encoder_geography.transform(input_data[['Geography']]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=[f'Geography_{geo}' for geo in onehot_encoder_geography.categories_[0]])
        input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

        # Encode gender
        input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_probab = prediction[0][0]

        return prediction_probab
    except Exception as e:
        raise ValueError(f"Error in processing: {str(e)}")