# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st

## Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    analyzer = SentimentIntensityAnalyzer()
    # Use the raw text (user input) for VADER sentiment analysis
    sentiment_score = analyzer.polarity_scores(user_input)  # Use raw input text
    
    # Get the compound score from VADER
    compound_score = sentiment_score['compound']
    
    # Determine sentiment based on the compound score
    sentiment = 'Positive' if compound_score >= 0.05 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment} (Score: {compound_score})')
else:
    st.write('Please enter a movie review.')