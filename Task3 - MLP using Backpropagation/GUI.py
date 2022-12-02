# Use streamlit for GUI
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie

st.set_page_config(page_title='Adaline', page_icon=':star:')


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


meditation_man = load_lottie(
   'https://assets5.lottiefiles.com/packages/lf20_O2ci8jA9QF.json')


st.header('Task3 - MLP Using Backpropagation')
st.write('---')

with st.container():
    l, r = st.columns(2)
    with l:
        st.subheader(':stars:Set Parameters:')
        learning_rate = st.number_input('Learning Rate: ')
        number_of_epochs = st.number_input('Number Of Epochs: ')
        num_of_hidden_layers = st.number_input('Number Of Hidden Layers: ')
        number_of_neurons = st.number_input('Number Of Neurons: ')
        Act_func = ('sigmoid', 'tanH')
        Activation_func = st.selectbox('Select Activation Function:', Act_func)
        bias = st.checkbox('Add Bias')
        st.write('##')

    with r:
        st_lottie(meditation_man, height=500, width=400, key='Man')


######## Implementation #######



##############################


if st.button('Train and Test Model'):
    pass
    
