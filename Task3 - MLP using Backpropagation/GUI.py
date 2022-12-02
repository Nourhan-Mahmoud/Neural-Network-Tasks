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
        num_iterations = st.number_input('Number Of Iterations: ')
        num_of_hidden_layers = st.number_input('Number Of Hidden Layers: ')
        number_of_neurons = st.number_input('Number Of Neurons: ')
        Act_func = ('sigmoid', 'tanH')
        Activation_func = st.selectbox('Select Activation Function:', Act_func)
        bias = st.checkbox('Add Bias')
        st.write('##')

    with r:
        st_lottie(meditation_man, height=500, width=400, key='Man')


######## Implementation #######


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn as sns
import math
# Loading data


train_fdata = pd.DataFrame(pd.read_csv("mnist_train.csv"))
test_fdata  = pd.DataFrame(pd.read_csv("mnist_test.csv"))
train_label = np.array(train_fdata["label"])
train_data= np.array(train_fdata.drop("label" , axis =1))
test_label=np.array(test_fdata['label'])
test_data=np.array(test_fdata.drop("label",  axis =1 ))
# Preprocessing


def encoding_labels(data):
    train_label_encoded=np.zeros((10,data.shape[0]))
    for ind in range (data.shape[0]):
        val=data[ind]
        for row in range (10):
            if (val==row):
                train_label_encoded[val,ind]=1
    return train_label_encoded
train_data =np.transpose(train_data)
train_label=encoding_labels(train_label)
test_data =np.transpose(test_data) 
test_label=encoding_labels(test_label)
print(train_data.shape ,train_label.shape , test_data.shape ,test_label.shape )
# Activation functions

def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def tanH(Z):
    return (np.tanh(Z)) 

# Derivative of activation functions
def sigmoid_backward(Z):    
    s = sigmoid(Z)
    dZ = s * (1-s)
    return dZ
def tanH_backward(Z):
    tanh = tanH(Z)
    dZ = (1 + tanh)*( 1- tanh) 
    return dZ
    
# Initialization
def initialize_parameters(num_of_hidden_layers, number_of_neurons ,num_of_epochs  ,input_layer, output_layer ,
                          Activation_func = "sigmoid" , Add_bias=True , eta = 0.01  ):
    #initialize parameters
    parameters = {}
    parameters["Activation_function"]=Activation_func
    parameters["learning_rate"] =eta
    parameters["Num_of_epochs"] = num_of_epochs
    parameters["Add_bias"] = Add_bias
    parameters["number_of_layers"] =num_of_hidden_layers+1
    
    #initialize weights
    weights ={}
    L = num_of_hidden_layers+2           # number of layers in the network
    layers = [input_layer]
    for i in range(num_of_hidden_layers):
        layers.append(number_of_neurons)
    layers.append(output_layer)
    for l in range(1, L):
        weights['W' + str(l)] = np.random.randn(layers[l] , layers[l-1])
        weights['b' + str(l)] = np.zeros(( layers[l] ,1))
    
    return parameters , weights
# Backpropogation Model
#<h4> Feedforward propagation</h4>


def feedforward_propagation(X, parameters , weights):
    caches = []
    A = X
    num_of_layers = parameters["number_of_layers"]+1             
    for l in range(1, num_of_layers):
        W ,b , activation = weights['W' + str(l)], weights['b' + str(l)],parameters["Activation_function"]
        Z = (np.dot(W,A)) +b
        cache = (A , W , b , Z)
        if activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "tanH":
            A = tanH(Z)
        caches.append(cache)               
    return A , caches 
#<h4> Backward Model</h4>


def activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    if activation == "tanH":
        dZ = dA* tanH_backward(Z) 
    elif activation == "sigmoid":
        dZ = dA* sigmoid_backward(Z)
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)  
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db



def L_model_backward(AL, Y, caches  , parameters ):
    #output layer update
    grads = {}
    L = parameters["number_of_layers"]  
    E = AL - Y
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp  = activation_backward(E, current_cache,parameters["Activation_function"])
    grads["dA"+str(L)]=dA_prev_temp
    grads["dW"+str(L)]=dW_temp
    grads["db"+str(L)]=db_temp
    # hidden layersupdate
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l +2)], current_cache,parameters["Activation_function"])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)]=db_temp
    return grads
#<h4> update parameters</h4>

def update_parameters(weights, grads, learning_rate ,parameters ):
    for l in range(parameters["number_of_layers"]):
        weights["W" + str(l+1)] =weights["W" + str(l+1)] + (learning_rate*grads["dW" + str(l+1)])
        if(parameters["Add_bias"]):
            weights["b" + str(l+1)] = weights["b" + str(l+1)] + (learning_rate*grads["db" + str(l+1)])
    return weights


def model(parameters , weights , X ,Y):
    for i in range(0, parameters["Num_of_epochs"]):
        # Implementing feedforward propagation
        A1 , caches = feedforward_propagation(X, parameters , weights)
        # Calculating error 
        L = len(caches)
        grads = L_model_backward(A1, Y, caches ,parameters)
        #updating parameters
        weights = update_parameters(weights, grads, parameters["learning_rate"] ,parameters) 
    return parameters , weights   

def fit(X, Y, learning_rate , num_iterations ,num_of_hidden_layers , number_of_neurons , Activation_func, Add_bias=True):
    parameters ,weights = initialize_parameters(num_of_hidden_layers, number_of_neurons ,num_iterations , X.shape[0],
    Y.shape[0] , Activation_func , Add_bias=True , eta = learning_rate)
    parameters ,weights  = model(parameters , weights , X ,Y)
    return parameters ,weights 


def compute_cost(A, Y):
    m = Y.shape[1]
    count =0
    for i in range(m):
        ind1 = -1
        for j in range(10):
            if Y[j][i]==1:
                ind1=j
                break
        ind2=-1
        maxi= -400
        for j in range(10):
            if A[j][i]>maxi:
                ind2=j
                maxi=A[j][i]
       # print(ind1 ,ind2,A[ind2][i] )
        if ind1 == ind2:
            count+=1      
    return count/m




##############################


if st.button('Train and Test Model'):
    parameters,weights = fit(train_data, train_label, learning_rate ,  int(num_iterations) , int(num_of_hidden_layers) , int(number_of_neurons) , Activation_func,bias) 
    a ,_= feedforward_propagation(test_data, parameters,weights) 
    st.write("Accuracy" ,compute_cost(a ,test_label) )
    
