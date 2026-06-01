import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# --------------------------
# CUSTOM CSS
# --------------------------

st.markdown("""
<style>

.main{
    background-color:#0e1117;
}

.title{
    text-align:center;
    font-size:42px;
    color:#00d4ff;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:white;
    margin-bottom:20px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background:#1f2937;
    color:white;
    font-size:22px;
}

.stButton>button{
    width:100%;
    background:#00d4ff;
    color:black;
    font-weight:bold;
    border-radius:10px;
    height:50px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------

st.markdown(
    "<div class='title'>🎬 Movie Review Sentiment Analysis System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning Based Sentiment Classification</div>",
    unsafe_allow_html=True
)

# --------------------------
# LOAD MODELS
# --------------------------

@st.cache_resource
def load_models():

    rnn = tf.keras.models.load_model(
        "simple_rnn_model.keras",
        compile=False
    )

    lstm = tf.keras.models.load_model(
        "lstm_model.keras",
        compile=False
    )

    gru = tf.keras.models.load_model(
        "gru_model.keras",
        compile=False
    )

    return rnn,lstm,gru

rnn_model,lstm_model,gru_model = load_models()

# --------------------------
# IMDB WORD INDEX
# --------------------------

word_index = imdb.get_word_index()

# --------------------------
# PREPROCESS FUNCTION
# --------------------------

def preprocess_review(review):

    review = review.lower()

    review = re.sub(
        r'[^a-zA-Z ]',
        '',
        review
    )

    words = review.split()

    sequence = []

    for word in words:

        if word in word_index:

            sequence.append(
                word_index[word] + 3
            )

    padded = pad_sequences(
        [sequence],
        maxlen=200
    )

    return padded

# --------------------------
# PREDICTION FUNCTION
# --------------------------

def predict(model, review):

    processed = preprocess_review(
        review
    )

    prob = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prob >= 0.5
        else "Negative"
    )

    confidence = (
        prob*100
        if prob >=0.5
        else (1-prob)*100
    )

    return sentiment,confidence,prob

# --------------------------
# MODEL SELECTOR
# --------------------------

selected_model = st.selectbox(

    "Select Model",

    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# --------------------------
# INPUT REVIEW
# --------------------------

review = st.text_area(

    "Enter your movie review here...",

    height=180
)

# --------------------------
# BUTTON
# --------------------------

if st.button("Analyze Review"):

    if review.strip()=="":

        st.warning(
            "Please enter a review."
        )

    else:

        if selected_model=="SimpleRNN":

            sentiment,confidence,prob = predict(
                rnn_model,
                review
            )

        elif selected_model=="LSTM":

            sentiment,confidence,prob = predict(
                lstm_model,
                review
            )

        else:

            sentiment,confidence,prob = predict(
                gru_model,
                review
            )

        st.markdown(
            f"""
            <div class='result-box'>
            Sentiment : <b>{sentiment}</b><br><br>
            Confidence : <b>{confidence:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        positive_prob = prob*100
        negative_prob = (1-prob)*100

        chart_df = pd.DataFrame({

            "Probability":[
                positive_prob,
                negative_prob
            ]

        },

        index=[
            "Positive",
            "Negative"
        ])

        st.subheader(
            "Probability Distribution"
        )

        st.bar_chart(chart_df)

# --------------------------
# COMPARE ALL MODELS
# --------------------------

st.markdown("---")

st.subheader(
    "Compare All Models"
)

if st.button("Compare Predictions"):

    if review.strip()=="":

        st.warning(
            "Please enter a review."
        )

    else:

        rnn_sent,rnn_conf,_ = predict(
            rnn_model,
            review
        )

        lstm_sent,lstm_conf,_ = predict(
            lstm_model,
            review
        )

        gru_sent,gru_conf,_ = predict(
            gru_model,
            review
        )

        comparison = pd.DataFrame({

            "Model":[
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],

            "Sentiment":[
                rnn_sent,
                lstm_sent,
                gru_sent
            ],

            "Confidence":[
                round(rnn_conf,2),
                round(lstm_conf,2),
                round(gru_conf,2)
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True
        )