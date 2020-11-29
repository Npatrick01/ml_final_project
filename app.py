import streamlit as st 
import tensorflow as tf
import joblib,os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer

    


model=tf.keras.models.load_model('ml_final2_model.h5')

st.title("")
	# st.subheader("ML App with Streamlit")
html_temp = """
	<div style="background-color:green;padding:10px">
	<h1 style="color:white;text-align:center;">Depressioin Detecting  App </h1>
	</div>

	"""
st.markdown(html_temp,unsafe_allow_html=True)
post_text = st.text_area("TYPE YOUR POST HERE","")
np.array(list(post_text))

if st.button("Classify"):
    max_words = 4000
    max_len = 400
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(post_text)
    sequences = tokenizer.texts_to_sequences(post_text)
    post_text = pad_sequences(sequences, maxlen=max_len)
    test_prediction =model.predict(post_text)
    if np.around(test_prediction, decimals=0)[0][0] == 1.0:
        st.write('You are depressed.Please visit the counselor')
    else:
        st.write("You are not depressed")

        