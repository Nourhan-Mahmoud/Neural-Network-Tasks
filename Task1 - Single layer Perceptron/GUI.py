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
        
    with r:
        st_lottie(penguin_animation, height=300,width=400,key = 'Cute Penguin')



#### Implementaion#### 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def signum(x):
    if x >= 0:
        return 1
    else:
        return -1
def perceptron(x, w, b):
    return signum(np.dot(w, x) + b)
def normalizeData(data : pd.DataFrame):
    data = np.abs(data - data.mean()) / data.std()
    return data
data = pd.read_csv('penguins.csv')


# PreProcessing
data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
      'body_mass_g']] = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                              'body_mass_g']].apply(lambda x: normalizeData(x))
data.gender.fillna(data.gender.mode()[0],inplace=True)
data.gender.unique()


#encode male to 0 and female to 1
gender_dict ={"male":1,"female":0}
data.gender = data.gender.apply(lambda x: gender_dict[x])


# Train Test Split
def splitData(data):
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[:int(len(data)*0.2)]
    test_data = data[int(len(data)*0.3):]
    return train_data,test_data

train_df,test_df = splitData(data)


# Model
def train(data,epoch,learning_rate,feature1,feature2,class1,class2,isBias):
    w = np.array([0, 0])
    b = 0
    data=data[[feature1,feature2,"species"]]
    data = data[(data.species == class1) | (data.species == class2)]
    print(data.species.unique())
    data = data.reset_index(drop=True)
    data[[feature1,feature2]] = data[[feature1,feature2]].apply(lambda x: normalizeData(x))
    species_dict = {class1:1,class2:-1}
    data.species = data.species.apply(lambda x: species_dict[x])
    for _ in range(epoch):
        for index, row in data.iterrows():
            x = np.array(row)
            y = x[2]
            x = x[0:2]
            if y * perceptron(x, w, b) <= 0:
                w = w + learning_rate * y * x
                if isBias:
                    b = b + learning_rate * y
    return w,b



def test(data,feature1,feature2,class1,class2,w,b):
    data=data[[feature1,feature2,"species"]]
    data = data[(data.species == class1) | (data.species == class2)]
    data = data.reset_index(drop=True)
    data[[feature1,feature2]] = data[[feature1,feature2]].apply(lambda x: normalizeData(x))
    species_dict = {class1:1,class2:-1}
    data.species = data.species.apply(lambda x: species_dict[x])
    correct = 0
    for index, row in data.iterrows():
        x = np.array(row)
        y = x[2]
        x = x[0:2]
        if y * perceptron(x, w, b) > 0:
            correct += 1
    return correct/len(data)



ww = train(train_df,int(number_of_epochs),learning_rate,option1_feature,option2_feature,option1_class,option2_class,bias)


if st.button('Train and Test SLP Model'):
    st.write('weights : ',ww) 