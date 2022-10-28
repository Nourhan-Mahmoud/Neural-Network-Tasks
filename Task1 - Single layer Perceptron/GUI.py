# Use streamlit for GUI
import streamlit as st
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title = 'SLP' , page_icon = ':star:')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
       return None
    return r.json()

penguin_animation = load_lottie('https://assets4.lottiefiles.com/packages/lf20_1cgsfbmb.json')


st.header('Task1 - Single Layer Perceptron On Penguin Dataset')
st.write('---')
with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        st.subheader(':stars:Select 2 Features:')
        features = ('bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g')
        option1_feature = st.selectbox(
            'Select First Feature:',features)
        st.write('You selected:', option1_feature)
        new_features = tuple(item for item in features if item != option1_feature)
        option2_feature = st.selectbox(
            'Select Second Feature:',new_features)
        st.write('You selected:', option2_feature)  
        

    with right_col:
        st.subheader(':stars:Select 2 Classes: ')
        classes = ('Adelie','Gentoo','Chinstrap')
        option1_class = st.selectbox(
            'Select First Class:',classes)
        st.write('You selected:', option1_class)
        new_classes = tuple(item for item in classes if item != option1_class)
        option2_class = st.selectbox(
            'Select Second Class:',new_classes)
        st.write('You selected:', option2_class)
 
st.write('####')
with st.container():
    l,r = st.columns(2)    
    with l:
        st.subheader(':stars:Set Other Parameters:')
        learning_rate= st.number_input('Learning Rate: ')   
        number_of_epochs= st.number_input('Number Of Epochs: ')
        bias = st.checkbox('Add Bias') 
        st.write('##') 
        if st.button('Train and Test SLP Model'):
            st.write('Done!') 
        
    with r:
        st_lottie(penguin_animation, height=300,width=400,key = 'Cute Penguin')

