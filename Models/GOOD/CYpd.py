import numpy as np
import tensorflow as tf
import pandas as pd 
from tqdm import tqdm

def loss_func(y,y_pred):
    return tf.reduce_mean(tf.square(y-y_pred))


dataset = pd.read_csv("D:/fng/fun_n_games/DATASETS/train.csv").to_numpy()
x = dataset[:,2:3]
y = dataset[:,1:2]
######################## ###### ##### ##matching encoded integers to text## ##### ###########################

#### mismatch between datatypes means no auto diff, first form a vocabulary and vectorize the words####
layer = tf.keras.layers.TextVectorization()
layer.adapt(y)
y_vector = layer(y)
y_vector = tf.cast(y_vector,dtype='float32')

x = np.asfarray(x)



class inference_test(tf.Module):
    def __init__(self):
        self.k1 = tf.Variable(tf.random.uniform([163]))
        self.k2 = tf.Variable(tf.random.uniform([163]))
        self.k3 = tf.Variable(tf.random.uniform([163]))
        
    def __call__(self,x):
        return tf.nn.softmax((x**2)*self.k1+(self.k2*x)+self.k3)

model = inference_test()

loss_data = []
epochs=3
for e in tqdm(range(epochs)):
    with tf.GradientTape() as tape:
        loss_value = loss_func(x,model(y_vector))
    grads = tape.gradient(loss_value, model.variables)      
    for g,v in zip(grads,model.variables):
        v.assign_sub(g)
    loss = loss_func(x,y_vector)
    loss_data.append(loss)
    

