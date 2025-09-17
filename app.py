import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simaple_rnn_imdb.h5')

# Function to decode review (optional, not used in UI)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# ─────────────────────────────────────────
# 🎨 Streamlit App UI
# ─────────────────────────────────────────

# Page Config
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>🎬 IMDB Movie Review Sentiment Analysis</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Enter a movie review below to classify it as <strong style='color:green;'>Positive</strong> or <strong style='color:red;'>Negative</strong>.</p>",
    unsafe_allow_html=True,
)

# Input box
user_input = st.text_area("📝 Your Movie Review", height=150, placeholder="Type or paste a review here...")

# Button
if st.button("🔍 Classify Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before classifying.")
    else:
        sentiment, score = predict_sentiment(user_input)

        # Display results
        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)

        if sentiment == "Positive":
            st.success(f"✅ Sentiment: **{sentiment}**")
        else:
            st.error(f"❌ Sentiment: **{sentiment}**")

        st.info(f"📊 Prediction Score: `{score:.4f}`")

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit and Keras</p>",
    unsafe_allow_html=True,
)
