import tensorflow as tf 
import numpy as np

x,y = np.load("BS_x.npy",allow_pickle=True), np.load("BS_y.npy",allow_pickle=True)
d_mod = np.shape(x[0])

def loss_function(y,y_pred):
    return abs(y - y_pred)

class linear_ts(tf.Module):
    def __init__(self):
        self.l1 = tf.Variable(tf.random.uniform(d_mod))
        self.l2 = tf.Variable(tf.random.uniform(d_mod))
        
    def __call__(self,x):
        return tf.nn.relu(tf.square(self.l1 *x) * self.l2)


model = linear_ts()

sample_count = 1000
loss_data = []
for i in range(sample_count):
    with tf.GradientTape() as tape:
        loss_value = loss_function(y[i], model(x[i]))
    grads = tape.gradient(loss_value, model.variables)
    for g,v in zip(grads,model.variables):
        v.assign_sub(g)
    if i % 10 == 0:
        print(loss_value.numpy())





