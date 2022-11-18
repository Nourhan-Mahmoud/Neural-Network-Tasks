# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
def activation_function(prediction):
    if prediction >= 0:
        return 1
    return 0

# %%
def predict(x, weights, bias):
    prediction = np.dot(weights, x.reshape(-1,1)) + bias
    print("prediction: ", prediction)
    return activation_function(prediction)

# %%
def forward_propagation(x, y, weights, bias): 
    y_pred = predict(x, weights, bias)
    loss = (y_pred - y)**2   
    d_loss = 2*(y_pred - y)
    
    return y_pred, loss, d_loss

# %%

def backpropagation(x, d_loss):
    partial_derivates = list()
    for feature_value in x:
        partial_derivates.append(d_loss*feature_value)
        
    return partial_derivates 

# %%
def optimize_perceptron(x, y, learning_rate):
    epoch = 0
    error = 999
    weights = np.random.rand(x.shape[1])
    bias = np.random.rand()
    
    errors = list()
    epochs = list()
    
    # Loop until stop conditions are met
    while (epoch <= 1000) and (error > 9e-4):
        
        loss_ = 0
        # Loop over every data point
        for i in range(x.shape[0]):
            
            # Forward Propagation on each data point
            y_pred, loss, d_loss = forward_propagation(x[i], y[i], weights, bias)

            # Backpropagation
            partial_derivates = backpropagation(x[i], d_loss)
            
            # Learn by updating the weights of the perceptron
            weights = weights - (learning_rate * np.array(partial_derivates))

        # Evaluate the results
        for index, feature_value_test in enumerate(x):
            y_pred, loss, d_loss = forward_propagation(feature_value_test, y[index], weights, bias)
            loss_ += loss

        errors.append(loss_/len(x))
        epochs.append(epoch)
        error = errors[-1]
        epoch += 1

        print('Epoch {}. loss: {}'.format(epoch, errors[-1]))
    
    return weights, bias, errors

# %%
def normalizeData(data : pd.DataFrame):
    data = np.abs(data - data.mean()) / data.std()
    return data

# %%
data = pd.read_csv('penguins.csv')

# %%
data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
      'body_mass_g']] = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                              'body_mass_g']].apply(lambda x: normalizeData(x))

# %%
data.gender.fillna(data.gender.mode()[0],inplace=True)
data.gender.unique()

# %%
gender_dict ={"male":1,"female":0}
data.gender = data.gender.apply(lambda x: gender_dict[x])

# %%
import warnings
warnings.filterwarnings("ignore")
def splitData(data,class1,class2):
    data = data.sample(frac=1,random_state=45).reset_index(drop=True)
    #randomly select 80% of the data
    class1_data = data[data.species == class1]
    class2_data = data[data.species == class2]
    test_data = pd.concat([class1_data[:int(len(class1_data)*0.3)],class2_data[:int(len(class2_data)*0.3)]])
    train_data = pd.concat([class1_data[int(len(class1_data)*0.3):],class2_data[int(len(class2_data)*0.3):]])
    print(len(train_data),len(test_data))
    return train_data,test_data

# %%
train_df,test_df = splitData(data,"Adelie","Chinstrap")

# %%

x= train_df[['bill_length_mm', 'bill_depth_mm']]
y = train_df[["species"]]
y = y.replace({"Adelie":0,"Chinstrap":1})
optimize_perceptron(x.values,y.values,0.01)


