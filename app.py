import streamlit as st
import pickle
import re
import string
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

# -----------------------------
# Load Model & Tokenizer
# -----------------------------

@st.cache_resource
def load_model_and_tokenizer():
    
    model = load_model("sentiment_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

max_len = 100


# -----------------------------
# Text Cleaning Function
# -----------------------------

def clean_text(text):

    text = text.lower()
    
    text = re.sub(r"http\S+", "", text)
    
    text = re.sub(r"@\w+", "", text)
    
    text = re.sub(r"#\w+", "", text)
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    return text


# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(review):

    review = clean_text(review)

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(sequence, maxlen=max_len, padding="post")

    prediction = model.predict(padded)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return sentiment, prediction


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🛍️ Product Review Sentiment Analysis")

st.write(
    "This app predicts whether a product review expresses **Positive** or **Negative** sentiment using a Deep Learning model."
)

review = st.text_area(
    "Enter your product review:",
    height=150,
    placeholder="Example: This product is amazing and works perfectly!"
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")

    else:

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction Result")

        if "Positive" in sentiment:
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

        st.write(f"Confidence Score: **{confidence:.2f}**")


st.markdown("---")

st.caption("Built with Streamlit | Deep Learning Sentiment Analysis")