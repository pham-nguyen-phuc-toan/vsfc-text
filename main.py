import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positve'}

st.title('Sentiment analysis from Vietnamese students’ feedback')

input = open('lrc_vsfc.pkl', 'rb')
model = pkl.load(input)

st.header('Write a feedback')
txt = st.text_area('', '')
st.write(txt)

if txt != '':
    if st.button('Predict'):
        feature_vector = txt
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
else:
        st.header('Result')
        st.text('There is no text here')
