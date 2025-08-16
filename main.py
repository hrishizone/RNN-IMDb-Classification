import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Load the IMDb word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

## Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

## Decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Preprocess input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction
def predict_review(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

## Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review:")

if st.button("Predict"):
    sentiment, confidence = predict_review(user_input)

    # Choose color based on sentiment
    if sentiment == "positive":
        color = "#4CAF50"  # green
        emoji = "🌟"
    else:
        color = "#F44336"  # red
        emoji = "💔"

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            color: white;
            font-size: 20px;">
            <h3 style="margin:0;">{emoji} Sentiment: {sentiment.capitalize()}</h3>
            <p style="margin:5px 0 0 0;">Confidence: <b>{confidence:.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Please enter a review to analyze.")
    