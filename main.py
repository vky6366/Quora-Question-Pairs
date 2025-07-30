import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
w2v_model = Word2Vec.load("Utils\w2v.model")
import Utils.process as p  # contains query_point_creator()
# from Utils.model import create_model  # optional if you want to recreate instead of loading

# Load pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model("Utils/duplicate_question_model.h5")

model = load_trained_model()

st.title('🔍 Duplicate Question Detector')

q1 = st.text_input('Enter Question 1:')
q2 = st.text_input('Enter Question 2:')

if st.button('Find'):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        query = p.query_point_creator(q1, q2, w2v_model)

        # Ensure correct shape
        prediction = model.predict(query)[0][0]

        if prediction <= 0.5:
            st.success('✅ Duplicate')
        else:
            st.info('❌ Not Duplicate')
